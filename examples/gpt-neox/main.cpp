#include "ggml/ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// default hparams (Pythia 70m)
struct gptneox_hparams {
    int32_t n_vocab = 50304;
    int32_t n_ctx   = 2048;
    int32_t n_embd  = 512;
    int32_t n_head  = 8;
    int32_t n_layer = 6;
    int32_t n_rot   = 16; // (n_embd / n_head) * 0.25
    int32_t f16     = 1;
};

struct gptneox_layer {
    // normalization
    struct ggml_tensor * ln_in_w;
    struct ggml_tensor * ln_in_b;

    struct ggml_tensor * ln_attn_w;
    struct ggml_tensor * ln_attn_b;

    // attention
    struct ggml_tensor * c_attn_qkv_proj_w;
    struct ggml_tensor * c_attn_qkv_proj_b;

    struct ggml_tensor * c_attn_proj_w; // dense
    struct ggml_tensor * c_attn_proj_b;

    // ff
    struct ggml_tensor * c_mlp_h_to_4h_w;
    struct ggml_tensor * c_mlp_h_to_4h_b;

    struct ggml_tensor * c_mlp_4h_to_h_w;
    struct ggml_tensor * c_mlp_4h_to_h_b;
};

struct gptneox_model {
    gptneox_hparams hparams;

    struct ggml_tensor * embed_in; // position embedding

    std::vector<gptneox_layer> layers;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * embed_out; // language model head

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// load the model's weights from a file
bool gptneox_model_load(const std::string & fname, gptneox_model & model, gpt_vocab & vocab) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        // GPTNeoX has fewer items in its vocabulary than n_vocab,
        // so let's say it's probably okay for the real vocab to be smaller than n_vocab.
        if (n_vocab > model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d > %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats
    // in order to save memory and also to speed up the computation
    const ggml_type wtype = model.hparams.f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*ggml_type_size(GGML_TYPE_F32); // ln_f_g
        ctx_size += n_embd*ggml_type_size(GGML_TYPE_F32); // ln_f_b

        ctx_size += n_embd*n_vocab*ggml_type_size(wtype); // embed_in

        ctx_size += n_embd*n_vocab*ggml_type_size(wtype); // embed_out

        ctx_size += n_layer*(n_embd*ggml_type_size(GGML_TYPE_F32)); // ln_in_w
        ctx_size += n_layer*(n_embd*ggml_type_size(GGML_TYPE_F32)); // ln_in_b

        ctx_size += n_layer*(n_embd*ggml_type_size(GGML_TYPE_F32)); // ln_attn_w
        ctx_size += n_layer*(n_embd*ggml_type_size(GGML_TYPE_F32)); // ln_attn_b

        ctx_size += n_layer*(3*n_embd*n_embd*ggml_type_size(wtype));  // c_attn_qkv_proj_w
        ctx_size += n_layer*(3*n_embd*ggml_type_size(GGML_TYPE_F32)); // c_attn_qkv_proj_b

        ctx_size += n_layer*(n_embd*n_embd*ggml_type_size(wtype));         // c_attn_proj_w
        ctx_size += n_layer*(       n_embd*ggml_type_size(GGML_TYPE_F32)); // c_attn_proj_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_size(wtype));         // c_mlp_h_to_4h_w
        ctx_size += n_layer*(       4*n_embd*ggml_type_size(GGML_TYPE_F32)); // c_mlp_h_to_4h_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_size(wtype));         // c_mlp_4h_to_h_w
        ctx_size += n_layer*(         n_embd*ggml_type_size(GGML_TYPE_F32)); // c_mlp_4h_to_h_b

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_size(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_size(GGML_TYPE_F32); // memory_v

        ctx_size += (6 + 12*n_layer)*256; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size   = ctx_size,
            .mem_buffer = NULL,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.embed_in = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.embed_out = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        // map by name
        model.tensors["gpt_neox.embed_in.weight"] = model.embed_in;

        model.tensors["gpt_neox.final_layer_norm.weight"] = model.ln_f_g;
        model.tensors["gpt_neox.final_layer_norm.bias"]   = model.ln_f_b;

        model.tensors["embed_out.weight"] = model.embed_out;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_in_w           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_in_b           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.ln_attn_w         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_attn_b         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_attn_qkv_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, 3*n_embd);
            layer.c_attn_qkv_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);

            layer.c_attn_proj_w     = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_proj_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_mlp_h_to_4h_w   = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_h_to_4h_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);

            layer.c_mlp_4h_to_h_w   = ggml_new_tensor_2d(ctx, wtype,         4*n_embd,   n_embd);
            layer.c_mlp_4h_to_h_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // map by name
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.weight"]           = layer.ln_in_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.bias"]             = layer.ln_in_b;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.weight"]  = layer.ln_attn_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.bias"]    = layer.ln_attn_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.weight"] = layer.c_attn_qkv_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.bias"]   = layer.c_attn_qkv_proj_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.weight"]           = layer.c_attn_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.bias"]             = layer.c_attn_proj_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.weight"]         = layer.c_mlp_h_to_4h_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.bias"]           = layer.c_mlp_h_to_4h_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.weight"]         = layer.c_mlp_4h_to_h_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.bias"]           = layer.c_mlp_4h_to_h_b;
        }
    }

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            const size_t bpe = tensor->type == GGML_TYPE_I8 ? 1 : (ftype == 0) ? sizeof(float) : sizeof(ggml_fp16_t);

            if (nelements*bpe != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }

    fin.close();

    return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool gptneox_eval(
        const gptneox_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_rot;

    const int d_key = n_embd/n_head;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = { .n_threads = n_threads };

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    // fprintf(stderr, "[");
    // for (int num : embd_inp) {
    //     fprintf(stderr, " %d ", num);
    // }
    // fprintf(stderr, "]\n");

    // wte
    // Notation: [ne0, ne1] means (cols, rows)
    // [n_embd, N]
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.embed_in, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // input norm
        // [n_embd, N] --> [n_embd, N]
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = ln_in_w*cur + ln_in_b
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ln_in_w, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].ln_in_b, cur));
        }

        // attn
        // [n_embd, N] --> [3*n_embd, N]
        {
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_attn_qkv_proj_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_attn_qkv_proj_b, cur),
                    cur);
        }

        // reshape attn
        {
            // [3*n_embd, N] --> [3*n_embd/n_head, n_head, N]
            cur = ggml_reshape_3d(ctx0, cur, 3*d_key, n_head, N);
        }
        // self-attention
        {
            // [3*d_key, n_head, N] --> [d_key, n_head, N]
            // TODO: check if ggml_view_3d is correctly implemented
            // And for the curious, no, reshape then split is not equivalent to split then reshape.
            struct ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, d_key, n_head, N, cur->nb[1], 0*ggml_element_size(cur)*d_key);
            struct ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, d_key, n_head, N, cur->nb[1], 1*ggml_element_size(cur)*d_key);
            struct ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, d_key, n_head, N, cur->nb[1], 2*ggml_element_size(cur)*d_key);

            // rotary embeddings
            // `cpy` is used to make Qcur and Kcur contiguous
            Qcur = ggml_rope(ctx0, ggml_cpy(ctx0, Qcur, ggml_new_tensor(ctx0, Qcur->type, Qcur->n_dims, Qcur->ne)), n_past, n_rot, 0);
            Kcur = ggml_rope(ctx0, ggml_cpy(ctx0, Kcur, ggml_new_tensor(ctx0, Kcur->type, Kcur->n_dims, Kcur->ne)), n_past, n_rot, 0);

            // store key and value to memory
            if (N >= 1) {
                // Get "current" memory view for k and v
                // N*n_embd elements at offset (TODO: explain offset)
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                // Copy Kcur and Vcur to memory (cache) (and expand graph?)
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // FIXME: rope or permute before or after?
            // [n_embd, N] --> [n_embd/n_head, n_head, N] --> [n_embd/n_head, N, n_head]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        // ggml_rope(ctx0,
                            // ggml_cpy(ctx0,
                                Qcur,
                                // ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                            // n_past, n_rot, 0),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // FIXME: rope or permute before or after?
            // [n_embd, n_past+N] --> [n_embd/n_head, n_head, n_past+N] --> [n_embd/n_head, n_past+N, n_head]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        // ggml_rope(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            // n_past, n_rot, 1),
                        0, 2, 1, 3);

            // K * Q
            // --> [n_past+N, N, n_head]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            //KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            // TODO: maybe use a 3d attention mask instead? (unless this does that)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            // [n_embd, n_past+N] --> [n_embd/n_head, n_head, n_past+N] --> [n_past+N, n_embd/n_head, n_head]
            struct ggml_tensor * V_trans =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        1, 2, 0, 3);

            // KQV = transpose(V) * KQ_soft_max
            // [n_past+N, n_embd/n_head , n_head] * [n_past+N, N, n_head] --> [n_embd/n_head, N, n_head]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [n_embd/n_head, N, n_head] --> [n_embd/n_head, n_head, N]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [n_embd/n_head, n_head, N] --> [n_embd, N]
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (with bias)
            // [n_embd, n_embd] * [n_embd, N] --> [n_embd, N]
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_attn_proj_w,
                    cur);
            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_attn_proj_b, cur),
                    cur);
        }

        struct ggml_tensor * outAttn = cur;

        // post-attention layer norm
        {
            // note here we pass inpL instead of cur
            // [n_embd, N]
            cur = ggml_norm(ctx0, inpL);

            // cur = ln_attn_w*cur + ln_attn_b
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ln_attn_w, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].ln_attn_b, cur));
        }
        // feed-forward network
        // this is independent of the self-attention result, so it could be done in parallel to the self-attention
        {
            // [n_embd, 4*n_embd] * [n_embd, N] --> [4*n_embd, N]
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_mlp_h_to_4h_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_mlp_h_to_4h_b, cur),
                    cur);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            // cur = proj_w*cur + proj_b
            // [4*n_embd, n_embd] * [4*n_embd, N] --> [n_embd, N]
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_mlp_4h_to_h_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_mlp_4h_to_h_b, cur),
                    cur);
        }

        // self-attention + FF
        cur  = ggml_add(ctx0, outAttn, cur);

        // input for next layer
        // add residual
        // [n_embd, N]
        inpL = ggml_add(ctx0, cur, inpL);
    }

    // final layer norm
    {
        // [n_embd, N]
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    // lm_head
    // [n_embd, n_vocab] * [n_embd, N] --> [n_vocab, N]
    inpL = ggml_mul_mat(ctx0, model.embed_out, inpL);

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    // [n_vocab]
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

int main(int argc, char ** argv) {
    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/gpt-neox/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    gptneox_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!gptneox_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    printf("\n");

    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    gptneox_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!gptneox_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 0) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
