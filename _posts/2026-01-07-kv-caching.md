---
layout: post
title: "The uncompromising intro to KV caching"
subtitle: "A basic optimization for autoregressive generation with Transformers"
cover-img: /assets/img/posts/2026-01-07-kv-caching/cover_cropped.png
thumbnail-img: /assets/img/posts/2026-01-07-kv-caching/thumb.png
share-img: /assets/img/posts/2026-01-07-kv-caching/thumb.png
# gh-repo: daattali/beautiful-jekyll
# gh-badge: [star, fork, follow]
tags: [transformers, attention]
comments: true
author: Francesco Cariaggi
---

In this blog post, I will try my best to explain key-value caching, or KV caching in short, in a beginner-friendly way, but with enough technical depth to make it enjoyable for non-beginners too. The "uncompromising" part of the title refers to the intention of avoiding shortcuts or simplifications that may prevent the reader from fully understanding what KV caching is and why it is useful.

## Introduction
If you want to fully understand key-value caching, you better know what keys and values are in the first place. **Keys** and **values**, along with **queries**, are the building blocks of **attention**, a central component of the [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) architecture that powers all modern Large Language Models and more. Quoting the original [paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf):
> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

where the compatibility function is usually the dot product. The mathematical formulation of the informal description above is:

$$
(q_1\, q_2\, ...\, q_n)
\cdot
\left(
\begin{array}{c|c|c|c}
k_{11} & k_{21} & \cdots & k_{m1} \\
k_{12} & k_{22} & \cdots & k_{m2} \\
\vdots & \vdots & \ddots & \vdots \\
k_{1n} & k_{2n} & \cdots & k_{mn} \\
\end{array}
\right)
\cdot
\left(
\begin{array}{cccc}
v_{11} & v_{12} & \cdots & v_{1l} \\\hline
v_{21} & v_{22} & \cdots & v_{2l} \\\hline
\vdots & \vdots & \ddots & \vdots \\\hline
v_{m1} & v_{m2} & \cdots & v_{ml} \\
\end{array}
\right)
$$

where the **query** is the leftmost row vector, the **keys** are the _columns_ of the middle matrix, and the **values** are the _rows_ of the rightmost matrix. If you read the Transformer paper, however, you might be familiar with this alternative formulation:

$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

which is totally equivalent, except:
- rather than performing attention on individual query vectors, we do it on multiple queries at a time by organizing them as the _rows_ of a matrix $$Q$$
- the key vectors are normally organized as _rows_ of the matrix $$K$$, not columns, which is why you see $$K^T$$ in the formulation above
- the _attention weights_, i.e. the dot products between queries and keys, are usually normalized such that they sum up to 1 (via the $$\text{softmax}(\cdot)$$ function)
- the attention weights are scaled by $$\sqrt{d_k}$$, where $$d_k$$ is the dimensionality of the query vectors (we called it $$n$$ in our formulation). This is done to avoid issues when backpropagating gradients through the softmax (see the paper for more details)

Now the core question: what do queries, keys and values represent in practical scenarios? A very popular analogy circulating online relates a **query** to some _research_ you want to perform, **values** to _information sources_, and **keys** to _summaries_ that help you understand whether or not a given information source is relevant your research. Let's make an example: suppose you want to research on American postmodernism (this is your _query_). You go to the local library where you find countless book shelves, so you start looking. To determine whether a given book is worth leafing through, you read the author's name and the title on its spine (these are the _keys_). After stumbling into some irrelevant books, you finally come to the right section of the library where you find books from authors such as David Foster Wallace, Don DeLillo, and Thomas Pynchon. Since this is exactly what you were looking for, you decide to pick them up and read them (these are the _values_) to gain as much information as possible on American postmodernism.

While this analogy is nice and intuitive, it doesn't quite fit the use case you probably have in mind for attention. After all, when we talk about attention nowadays, we usually mention it in the context of sophisticated AI models that can generate realistic images and videos, speak with human-like voices, or act as personal assistants. For these use case, what is an intuitive interpretation of queries, keys and values?
Let's start from the name given to these models: **Transformers**. Ultimately, Transformers take an input sequence and "transform" it into an output sequence. As this sequence makes its way across the layers of the model, it gets "transformed" into a different one by mixing the information carried by each element of the sequence with that of the other elements. As an example, for text-only models, the _input sequence_ is a sequence of _word embeddings_, and mixing the information carried by each element of the sequence is a way to learn _contextual_ representations of words: for instance, the word "read" could refer to either the present or the past tense of the verb "to read" depending on the context in which it occurs. In this information mixing process, queries, keys and values play specific roles. The queries are the "information seekers", the keys are the "informers" and the values are the "information sources". Think of it as a role-playing game: the information seekers ($$Q$$) reach out to the informers ($$K$$) to consult them ($$QK^T$$) on how relevant the information they carry ($$V$$) is in relation to what they're interested in. Based on their answers, the seekers extract the appropriate amount of information from each source ($$(QK^T)V$$) and compile it into a final report.

## Key-value caching
Let's now introduce key-value caching by discussing what it is, why it is useful and how it is implemented.

### What is KV caching?
As the name implies, key-value caching, or simply KV caching, is a caching mechanism for key and value vectors. But why would we want to cache them, and in what situations? Starting from the latter, KV caching is used in a very specific sitation, namely while performing **inference** with an **autoregressive model**. If either of the two conditions are not met, KV caching does <u>not</u> apply. For example, KV caching is useless while training a model. Similarly, KV caching won't help you if you're performing inference with a bidirectional model like [BERT](https://arxiv.org/abs/1810.04805). Now don't worry, I'm not stating this as an absolute truth expecting you to accept it quietly (like some blog posts do). I will elaborate on _why_ that is in the [KV caching FAQs](#kv-caching-faqs) section later.

Now that we know _when_ KV caching is used, we are left with the question of _why_ we may want to use it. As it turns out, during an **autoregressive text generation** process, many calculations are repeated at each step. To see this, let's look at an example:

| Step number | Input tokens                          | Generated token |
|:-----------:|:--------------------------------------|:----------------|
| 1           | \<bos\>                               | Hi,             |
| 2           | \<bos\> Hi,                           | how             |
| 3           | \<bos\> Hi, how                       | can             |
| 4           | \<bos\> Hi, how can                   | I               |
| 5           | \<bos\> Hi, how can I                 | help            |
| 6           | \<bos\> Hi, how can I help            | you             |
| 7           | \<bos\> Hi, how can I help you        | today?          |
| 8           | \<bos\> Hi, how can I help you today? | \<eos\>         |
{: .table-full-width}

where \<bos\> and \<eos\> are special tokens indicating the beginning and end of the sequence, respectively. As you can see, we are indeed running some of the inputs through the model multiple times throughout the autoregressive text generation process. The special \<bos\> token is run through the model 8 times in a row, "Hi," is run 7 times, and so on.

To start building an intuition of how KV caching can avoid repeating certain computations, let's visualize what "running inputs through the model" actually means. The following is a graphical representation of a Transformer block, where residual connections and layer normalizations have been omitted to avoid cluttering (read from bottom to top):

<figure style="text-align:center;">
  <img src="/assets/img/posts/2026-01-07-kv-caching/zoom_in_layer_l.drawio.svg" alt="zoom_in_layer_l">
  <figcaption><em>Figure 1: Schematic of a Transformer layer.</em></figcaption>
</figure>

where $$h_{l,\,i \,(i=0,\dots,n)}$$, also known as the _hidden states_, are the output of the token embedding layer if $$l=1$$ (1st Transformer block) or the output of layer $$l - 1$$ if $$l>1$$.

So when is the right moment to cache our key and value vectors, and how does caching them help us avoid repeating computations? If you recall from the introductory section, the attention operation requires a query vector and a set of key and value vectors as inputs. "Set" is the keyword here: at _each_ step of the autoregressive text generation process, you need the key and value vectors for _all_ the input tokens seen so far, not just for the current one like in the case of the query vector. Imagine you have a rather deep model, say with 32 layers, and try to picture what conditions need to be met before you can run the attention computation in the 32nd layer during the 10th autoregressive text generation step. Because you need 1 query vector, 10 key vectors and 10 value vectors, you must have run the current input token through the 32 layers of your model (to get the 1 query vector) _as well as_ <u>all</u> the input tokens you've seen so far, which amount to 10, to get the key and value vectors for each of them.

Now here's the catch: what happens during the 11th step? To get the 23 vectors you need for attention (1 query, 11 keys, 11 values), again you must feed your model not just the current token (for the query), but also the entirety of the 11 tokens seen so far (for the keys and values), <u>including</u> the 10 you fed at the previous step! So the smart thing to do at step 10 would have been to _cache_ the key and value vectors obtained by running the 10 tokens through your model, so that you wouldn't have to do the same thing at step 11.

### Why KV cache is useful
From the previous section, we know that KV cache allows us to avoid redundant computations to get the key and value vectors for the past tokens. In this section, we'll have a deeper look at what these redundant computations are.

Let's start with a visual intuition by illustrating what calculations are performed during the 3rd step of a hypothetical autoregressive text generation within an attention layer in two different scenarios: one in which we _don't_ use a KV cache (top) and one in which we _do_ (bottom).

<figure style="text-align:center;">
  <img src="/assets/img/posts/2026-01-07-kv-caching/kv_caching.drawio.svg" alt="KV_caching">
  <figcaption><em>Figure 2: Attention computation without (top) and with (bottom) a KV cache during the 3rd step of an autoregressive text generation process.</em></figcaption>
</figure>

Note that the $$\text{softmax}(\cdot)$$ operation is omitted to avoid cluttering. As you can see, <u>without</u> a KV cache, we are forced to compute the full attention matrix ($$QK^T$$), _including_ the entries that will be effectively discarded due to causal masking (_essential_ in autoregressive text generation). On the other hand, <u>with</u> a KV cache, we only compute the last row by using the current query vector. Also, with a KV cache, we never compute the masked entries of the attention matrix because we always multiply the _current_ query with current and _past_ keys (but never _future_ keys).

_Why_ do we need to recompute the _whole_ attention matrix at each step though? Well, remember the table from the previous section showing how at each step we are feeding the model tokens generated in previous steps? If we compute attention naively, we have no choice but to carry over _all_ past tokens at each step, because <u>for each of them</u>, we need a hidden state $$h$$ in _every_ layer $$l$$ to compute the query, key and value vectors (see Figure 1). To be nitpicky, this is true for all the Transformer layers in your model <u>except the last one</u>. The reason is that after the last Transformer layer, all we are left to do is predict the next token (with the so-called _language modeling head_), and for that we only need the hidden state associated with the <u>current</u> token. This means that in the last Transformer layer, we can avoid computing the query vectors for the past tokens (but not the key and value vectors!), hence also the corresponding rows of the attention matrix (from the first to the penultimate).

So how to avoid carrying over all the past tokens and recomputing the whole attention matrix at each generation step? Ideally, we would like to deal only with the current token, since it's the one that will be used to predict the next one. As it turns out, KV caching allows us to do exactly this. By caching key and value vectors computed in each layer, we can reuse them for future generation steps. Here's how a Transformer layer changes after introducing a KV cache (read from bottom to top):

<figure style="text-align:center;">
  <img src="/assets/img/posts/2026-01-07-kv-caching/zoom_in_layer_l_with_cache.drawio.svg" alt="zoom_in_layer_l">
  <figcaption><em>Figure 3: Schematic of a Transformer layer with KV caching.</em></figcaption>
</figure>

If you compare this to Figure 1, you immediately notice one important difference: what flows in and out of the layer is no longer a _sequence_ of hidden states, but a _single_ hidden state. This is enabled by the fact that key and value vectors for the past tokens are retrieved from the newly introduced KV cache for the layer $$l$$, so we don't need to compute them from scratch from the hidden states of the past tokens. In fact, after computing the key and value vectors for the current token ($$k_{l,\,n}$$ and $$v_{l,\,n}$$, respectively), we simultaneously update the KV cache with them _and_ retrieve the key and value vectors for the past tokens, which we then use to compute attention.

At this point you might be wondering: does KV caching come free of charge? Is there any reason at all _not_ to use it? Unfortunately, it does come with a non-trivial disadvantage: memory overhead. Normally, while performing inference with a model, any intermediate output of its layers and sublayers is discarded after being fed to the next layer or sublayer in the pipeline. However, when using a KV cache, we have to retain some of these intermediate outputs in the form of key and value vectors, which occupy memory and add up across generation steps. To understand the entity of such memory overhead, let's look at an example. Suppose we have:
- a 24-layer model
- key and value vectors of type float16 with 128 dimensions
- 32 attention heads (multi-head attention)
- a conversation context of 4096 tokens

With this particular setting, the KV cache would occupy 24 * 2 * 16 * 128 * 32 * 4096 = 12,884,901,888 bits, which corresponds to 1.5 GiB. So for conversations where you have a long past context (think reasoning models or models proudly supporting 1M+ context length), the KV cache can quickly become the bottleneck in terms of memory.
On the flipside, let's not forget about the big advantage of KV caching, which is to avoid repeated computations and to make the attention computation linear in the sequence length rather than quadratic (we need to compute only the last row of the attention matrix), both in terms of memory and compute.

### How is KV caching implemented in code?

To exemplify how KV caching is implemented in code, let's look at the Hugging Face implementation (v5.0.0rc1) of [OPT](https://arxiv.org/abs/2205.01068), an autoregressive large language model by Meta. In particular, let's focus on the `forward()` method of the [`OPTAttention`](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/opt/modeling_opt.py#L103) class:

```python
class OPTAttention(nn.Module):

    [...]

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        [...]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            **kwargs,
        )

        [...]
```

Note how the attention computation is conceptually organized in 3 distinct phases:
1. Query, key and value vectors are computed for the _current_ token:

    ```python
    query_states = self.q_proj(hidden_states) * self.scaling
    [...]
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    ```
2. Past key and value vectors are fetched from the KV cache (`past_key_values`):

    ```python
    if past_key_values is not None:
        # save all key/value_states to cache to be re-used for fast auto-regressive generation
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, {"cache_position": cache_position}
        )
    ```
3. Attention is finally computed:

    ```python
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        ...,
    )
    ```

As an exercise, go back and compare this implementation to Figure 3 to see if you can match each variable in the code with its corresponding entity in the diagram.

### KV caching FAQs
In this section, I'll try to answer some of the most common questions around KV caching, which are usually a symptom of a lack of deep understanding caused by superficial explanations of KV caching:

**Q**: **Can I use KV caching in encoder models like [BERT](https://arxiv.org/abs/1810.04805)?** 

**A**: No, because encoder models use attention <u>without</u> causal masking during training, meaning that each token has both left (past) and right (future) context. This implies that if you were to run autoregressive text generation with them, you'd find it hard (in fact, impossible) to cache key and value vectors because at step $$n$$, the key and value vectors for the current token (and for all the previous one, for that matter) will be <u>different</u> from the key and value vectors for the same token at step $$n+1$$. The reason? At step $$n+1$$, the same token will have context from the $$(n+1)$$th token as well, so its key and value vectors will change, meaning those from step $$n$$ are completely useless.

**Q**: **Can I use KV caching during training?**

**A**: You don't need a KV cache during training. At training time, you have access to the _whole_ output sequence upfront and don't need to generate it one token at a time, hence there are no repeated computations being performed.

**Q**: **Why not cache queries too?**

**A**: Recall the definition of attention from the introductory section: mapping a query and a set of key-value pairs to an output. This tells us that to compute attention, you only need the _current_ query and not the _past_ queries, so there's no point in caching them. On the other hand, as mentioned several times in this blog post, we _do_ need past keys and values, so it makes sense to cache them.

**Q**: **Does a KV cache store attention weights ($$q_ik_j$$) too?**

**A**: By looking at Figure 2, you might be tricked into thinking that when using a KV cache, most entries of the attention matrix are not computed because they are cached. This is incorrect: the _real_ reason they are not computed is that they are not _needed_. Again, at each generation step, you compute attention using the _current_ query plus the current and _past_ keys and values, so you never compute any entry of the upper triangular portion of the attention matrix.

## Wrapping up

In this post, we've taken a deep dive into KV caching, one of the most important optimizations for autoregressive Transformer inference. We started by building a solid foundation with the basics of attention mechanisms, understanding how queries, keys, and values work together to enable contextual information processing in modern language models.

We then explored KV caching in detail: what it is, why it matters, and how it fundamentally changes the inference process. The key insight is that by caching the key and value vectors computed for previous tokens, we avoid the quadratic complexity that would otherwise make autoregressive generation prohibitively expensive for long sequences. Instead of recomputing the entire attention matrix at each generation step, we only need to compute the attention weights for the current token against all past tokens.

However, this optimization comes with a memory cost that scales linearly with sequence length and model depth. For large models with long contexts, the KV cache can consume significant GPU memory (potentially gigabytes for conversations spanning thousands of tokens).

Understanding KV caching is crucial for grasping how modern language models can generate coherent, contextually aware text at interactive speeds. Whether you're building applications with LLMs, researching model efficiency, or simply curious about how AI systems work under the hood, KV caching represents a fundamental concept that bridges the gap between theoretical attention mechanisms and practical, scalable inference systems.

<!-- Refs
https://huggingface.co/blog/not-lain/kv-caching
https://medium.com/@joaolages/kv-caching-explained-276520203249
https://neptune.ai/blog/transformers-key-value-caching
https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/#key-value_caching
-->
