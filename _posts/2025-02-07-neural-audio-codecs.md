---
layout: post
title: "Neural Audio Codecs & (Residual) Vector Quantization"
subtitle: "The technology behind State-of-the-Art Audio AI models"
cover-img: /assets/img/posts/2023-12-24-neural-audio-codecs/cover.png
thumbnail-img: /assets/img/posts/2023-12-24-neural-audio-codecs/thumb.png
share-img: /assets/img/posts/2023-12-24-neural-audio-codecs/thumb.png
# gh-repo: daattali/beautiful-jekyll
# gh-badge: [star, fork, follow]
tags: [deep-learning, audio, neural-audio-codecs]
comments: true
author: Francesco Cariaggi
---

In this blog post, I'll take you through two important concepts behind modern Audio AI models such as Google's [AudioLM](https://arxiv.org/abs/2209.03143) and [VALL-E](https://arxiv.org/abs/2301.02111), Meta's [AudioGen](https://arxiv.org/abs/2209.15352) and [MusicGen](https://arxiv.org/abs/2306.05284), Microsoft's [NaturalSpeech 2](https://arxiv.org/abs/2304.09116), Suno's [Bark](https://github.com/suno-ai/bark), Kyutai's [Moshi](https://kyutai.org/Moshi.pdf) and [Hibiki](https://arxiv.org/abs/2502.03382), and many more: Neural Audio Codecs and (Residual) Vector Quantization.

If you don't mind a short primer/refresher (depending on your prior knowledge) on data compression (needed before delving into the actual topics of this blog post), then just read this blog post from the beginning. Otherwise, if you are already confident with concepts like codecs and bitrate, feel free to skip over to [Neural Audio Codecs](#neural-audio-codecs).

## Introduction

Did you ever wonder how multimedia files (music, videos, etc.) are efficiently stored in your PC? Have you got any clue as to how they can be transmitted in real time over the internet, for instance during videocalls?
Even if you don't know the details, you've probably guessed there must be some sort of _compression_ going on at some point. If there was no compression involved, your files would be pretty damn large (ever tried to extract audio tracks from a CD?) and the internet traffic would be much bigger than it currently is.

Let's take audio files as an example. Analog audio is converted to digital form by means of [Pulse-code modulation (PCM)](https://en.wikipedia.org/wiki/Pulse-code_modulation), which simply amounts to sampling the amplitude of analog audio at regular intervals or, equivalently, with a certain _sampling rate_, and then _quantize_ said values to the nearest value within a discrete range, for example 24-bit integers. Now suppose we want to digitize the song [Ode to My Family](https://www.youtube.com/watch?v=Zz-DJr1Qs54) by The Cranberries (great song, I know), which has a duration of 4 minutes and 32 seconds, or equivalently 272 seconds. Employing a sampling rate of 44.1kHz, our digitized audio will consist of 272 * 44100 = 11995200 samples, each encoded as a 24-bit integer. In this case, the size of the resulting file would be 11995200 * 24 = 287884800 bits = 35985600 bytes = over 34 MiB 🤯.

Can you imagine fitting songs *that* big in the kind of pocket-size music players we used to have 15+ years ago that only had a few hundred MiB of storage? Similarly, can you imagine streaming 30-50 MiB of data for *each* song you listen to nowadays on Spotify when you're on a limited internet data plan of a handful of gigabytes? In practice, thanks to compression, standard music tracks seldom exceed 2-3 MiB in size and sometimes are even smaller than 1 MiB.

## Codecs
Now we _know_ for a fact that multimedia files get compressed before being stored in a given device or transmitted over the internet, so let's see _how_ the compression takes place. As it turns out, we have software tools called **codecs** (portmanteau of coder/decoder) which serve precisely this purpose. You might not realize, but you have been dealing with codecs all the time: do [MP3](https://en.wikipedia.org/wiki/MP3) and [JPEG](https://en.wikipedia.org/wiki/JPEG) ring a bell? The former is a popular audio codec, whereas the latter is commonly used to compress images.

### Compression parameters
When thinking about compression, a couple of questions arise naturally:
1. How small can I make a file (without making it absolute trash)?
2. How does the compressed file compare to the original file?

The two parameters that help us answer the questions above are **bitrate** and **perceptual quality**.

The bitrate refers to the amount of bits required to encode a "unit" of data. For instance, in the case of audio codecs, said unit of data corresponds to 1 second of audio, hence the bitrate is expressed in _bits per second_ (bps). On the other hand, in the case of image codecs, a unit corresponds to 1 pixel, therefore the bitrate is expressed in _bits per pixel_ (bpp).
Perceptual quality, on the other hand, can be measured either with _objective_ metrics (such as [PESQ](https://en.wikipedia.org/wiki/Perceptual_Evaluation_of_Speech_Quality) and [STOI](https://ieeexplore.ieee.org/document/5495701) for audio) or via _subjective_ evaluations involving human experts.
A good codec aims to _minimize_ the bitrate while _maximizing_ the perceptual quality of the compressed data.

### Codecs categorization
Codecs can be categorized along two orthogonal dimensions:
- **lossy** vs **lossless**
- **generic** vs **content-aware** (I made this term up)

**Lossy** codecs, as the name suggests, are codecs that give up part of the original information to achieve a larger compression rate. Examples of lossy codecs are [MP3](https://en.wikipedia.org/wiki/MP3) for audio and [JPEG](https://en.wikipedia.org/wiki/JPEG) for images.
**Lossless** codecs, on the other hand, are codecs that can retain _all_ the original information while still being able to shrink the data a little bit. Examples of lossless codecs are [FLAC](https://en.wikipedia.org/wiki/FLAC) for audio and [PNG](https://en.wikipedia.org/wiki/PNG) for images.

<div class="box-hands-on" markdown="1">
🙌 **Hands-on**: proof that FLAC is a lossless audio codec (requires `ffmpeg`)

**Step 1**: Get a WAV file off the internet (or find one in your machine)
```sh
wget https://samples-files.com/samples/audio/wav/sample-file-3.wav -O audio-original.wav
```
**Step 2**: Compress it using `ffmpeg` and the FLAC codec
```sh
ffmpeg -y -i audio-original.wav -acodec flac audio-compressed.flac
```
**Step 3**: Check the compressed file is indeed smaller than the original one
```sh
ls -lh audio-original.wav audio-compressed.flac
```
**Step 4**: Use `ffmpeg` to decompress the FLAC file back to WAV
```sh
ffmpeg -y -i audio-compressed.flac audio-decompressed.wav
```
**Step 5**: Compare the original and decompressed files
```sh
diff -s <(ffmpeg -i audio-original.wav -f md5 - 2>/dev/null | grep MD5) <(ffmpeg -i audio-decompressed.wav -f md5 - 2>/dev/null | grep MD5)
```
**Note**: the reason why we don't run `diff` on the audio files directly is because their metadata might differ. Instead, we compare the MD5 checksums of their _contents_ (i.e. the actual audio tracks), which is what we're really interested in.
</div>


**Generic** codecs aim to reduce the size of the data without making any assumption on the nature of their inputs. **Content-aware** codecs, instead, rely on additional assumptions on the input data that allow them to achieve a better tradeoff between bitrate and perceptual quality. For example, [Speex](https://en.wikipedia.org/wiki/Speex) is an audio codec specifically designed and tuned to encode/decode human speech, hence it might not work very well, say, for music.

## Neural Audio Codecs {#neural-audio-codecs}
In order to achieve the best possible tradeoff between compression rate and perceptual quality of the reconstructed data, traditional audio codecs require careful design using hand-engineered Signal Processing techniques. In situations like this, it is natural to wonder if we can have a neural network *learn* to perform such a complex task from data. As you have probably guessed, the answer is yes 😎

*Neural audio codecs* are neural networks that try to learn how to reconstruct an audio signal given a compressed representation of it. If you're familiar with [Autoencoders](https://en.wikipedia.org/wiki/Autoencoder), this problem formulation won't sound new to you. Unsurprisingly, like autoencoders, neural audio codecs are made of an *encoder* and a *decoder*. The encoder takes a raw audio waveform as input and outputs a compressed representation of it. On the other hand, the decoder is fed the same compressed representation and is tasked with reconstructing the original audio waveform. Although quite simplistic, this description of how a neural audio codec works reveals an important fact about how such models can be trained. In particular, you might have noticed that no human supervision (i.e. labels) is needed to train neural audio codecs! As a matter of fact, the network is essentially requested to learn the identity function: given an audio waveform $$X$$, output a reconstructed version $$\hat{X}$$ such that $$\hat{X} = X$$ (in practice, we make it so that the two are as close as possible).

Soooo... basically we're out here racking our brains to train a sophisticated neural architecture to... learn the identity function?! 😅 That's right, but the point is that we're not interested in the network's output, but rather in the encoder's output, namely the *learned* compressed representation of the input waveform. But how do we obtain that, and how does it even look like? In this blog post, we'll learn about a procedure called **Residual Vector Quantization** adopted by state-of-the-art neural audio codecs such as Google's [SoundStream](https://arxiv.org/abs/2107.03312), Meta's [EnCodec](https://arxiv.org/abs/2210.13438), and Kyutai's [Mimi](https://kyutai.org/Moshi.pdf).


## (Residual) Vector Quantization {#residual-vector-quantization}

Broadly speaking, quantization is the process through which a continuous representation of a signal is mapped to a discrete space. For instance, as I mentioned at the very beginning of this blog post, Pulse-code modulation is a form of quantization.

More specifically, vector quantization (VQ) is a method for quantizing a vector of real numbers by means of a so-called *codebook*, that is a fixed-size collection of $$n$$ vectors that can be used to approximate any other vector. Formally, given a vector $$V_q$$ to be quantized and a codebook $$C = \{V_C^1, ..., V_C^n\}$$, we can obtain a quantized version of $$V_q$$ as:

$$\tilde{V}_q = \text{quant}(C, V_q) \stackrel{\text{def}}{=} \mathop{\mathrm{argmax}}\limits_i \,\, \text{s}(V_q, V_C^i)$$

where $$\text{s}(V, W)$$ is a function measuring the *similarity* between vectors $$V$$ and $$W$$. For example, a very simple $$s$$ could be the [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between $$V$$ and $$W$$:

$$\text{s}(V,W) = \frac{V\cdot W}{\lVert V \rVert \lVert W \rVert}$$

Since the codebook has a fixed size, it is inevitable for potentially quite different vectors $$V^1$$ and $$V^2$$ to be approximated with the same codebook vector $$V_C^i$$ ([Pigeonhole principle](https://en.wikipedia.org/wiki/Pigeonhole_principle)), leading to information loss after the quantization. This is where Residual Vector Quantization (RVQ) shines: what if we refined the approximation by first computing the approximation error, which is itself a vector, then approximating the approximation error (😵‍💫) by means of a second codebook? And what if we repeated this process again and again? 🤯

Now that we understand the intuition behind RVQ, let's get a little formal 🤵‍♂️ Given:
- a vector $$V_q$$ to be quantized
- a set of $$m$$ codebooks $$C_1 = \{V_{C_1}^1, ..., V_{C_1}^n\}$$, ..., $$C_m = \{V_{C_m}^1, ..., V_{C_m}^n\}$$

we can compute:

$$
\tilde{V}_q^1 = \text{quant}(C_1, V_q)
$$

$$
\tilde{V}_q^2 = \text{quant}(C_2, V_q - \tilde{V}_q^1)
$$

$$
\tilde{V}_q^3 = \text{quant}(C_3, V_q - \tilde{V}_q^1 - \tilde{V}_q^2)
$$

$$ \vdots $$

$$
\tilde{V}_q^m = \text{quant}(C_m, V_q - \sum_{j=1}^{m-1} \tilde{V}_q^j)
$$

If you're more of a visual learner, the same process is illustrated in the picture below:

![RVQ](/assets/img/posts/2023-12-24-neural-audio-codecs/rvq.svg)

Finally, we can obtain an approximation for $$V_q$$ as:

$$\tilde{V}_q = \sum_{j=1}^m \tilde{V}_q^j$$

This last bit of insight isn't necessarily obvious, so let's devote a little more time to it. In particular, let's focus on very last expression:

$$
\tilde{V}_q^m = \text{quant}(C_m, V_q - \sum_{j=1}^{m-1} \tilde{V}_q^j)
$$

Since quantization is really nothing fancier than an *approximation*, we can rewrite it as:

$$
\tilde{V}_q^m \approx V_q - \sum_{j=1}^{m-1} \tilde{V}_q^j
$$

now all we need to do is add $$\sum_{j=1}^{m-1} \tilde{V}_q^j$$ to both sides of the equation, which results in:

$$
V_q \approx \tilde{V}_q^m + \sum_{j=1}^{m-1} \tilde{V}_q^j = \sum_{j=1}^{m} \tilde{V}_q^j
$$

So while in the case of VQ the probability of two vectors $$V^1$$ and $$V^2$$ being approximated with the same codebook vector $$V_C^i$$ was $$1 / n$$, said probability shrinks to $$(1 / n)^m$$ in the case of RVQ.
At this point you might wonder: couldn't you simply use plain VQ with a larger codebook? By employing a codebook with $$n^m$$ vectors, the probability of "collision" would also be $$(1 / n)^m$$. Seems reasonable, right? Well, let me explain why RVQ is still a better choice than VQ with a larger codebook in this case.

Suppose all our vectors, i.e. the vectors to be quantized as well as codebook vectors, have $$p$$ elements each. If we were to use a codebook of size $$n^m$$, we would be dealing with $$p\cdot n^m$$ different numbers that need to be stored in memory. With RVQ, on the other hand, we would need only $$p\cdot n\cdot m$$ different numbers.
Also, remember I said this RVQ wizardry is used by state-of-the-art neural audio codecs? What if I told you these "numbers" are nothing but the learnable parameters of a neural network? Let's pick some arbitrary yet reasonable values for $$n$$, $$m$$ and $$p$$: let's say $$n = 512$$, $$m = 8$$ and $$p = 2048$$. If we decide to go for plain VQ, we would have to learn $$p\cdot n^m = 2048\cdot 512^8 = 9671406556917033397649408$$ parameters. Using RVQ, on the other hand, we would have to learn just $$p\cdot n\cdot m = 2048\cdot 512\cdot 8 = 8388608$$ parameters.

## Neural Audio Codecs in state-of-the-art audio AI models

Now that we learned how Neural Audio Codecs and RVQ work, you might be left with one last doubt: what's the connection between them and state-of-the-art AI models for tasks like Text-to-Speech and Speech-to-Speech translation? Sure, they can compress audio efficiently with minimal loss in quality, but how is that relevant?

As it turns out, RVQ-based Neural Audio Codecs can serve as the equivalent of text tokenizers for audio, with a small difference owing to their residual nature: while text tokenizers turn text sequences into integers representing indices of *tokens* in a vocabulary, RVQ-based Neural Audio Codecs turn audio sequences into *vectors* of integers with $$m$$ elements, where $$m$$ is the number of codebooks. The $$i$$-th element of each vector represents the index of $$\tilde{V}_q^i$$ in the codebook $$C_i$$.
Here are plausible outputs of some hypothetical text tokenizer and RVQ (<u>just the encoder part</u>):

$$
\text{tokenize}(\unicode{x201C}\text{Hello, world!"}) = [123092, 23234, ..., 892558] \in \mathbb{N}^p
$$

$$
\begin{align}
  \text{rvq_encoder}(🔊) &= [
    \begin{pmatrix}
         345 \\
         981 \\
         \vdots \\
         1012
       \end{pmatrix},
    \begin{pmatrix}
         623 \\
         514 \\
         \vdots \\
         991
       \end{pmatrix},
       ...,
    \begin{pmatrix}
         401 \\
         240 \\
         \vdots \\
         812
       \end{pmatrix}
    ] \in \mathbb{N}^{m \times q}
\end{align}
$$

So what can we do with audio tokenizers? Well, exactly the same things we can do with regular tokenizers, for instance training *language models* on their output. [AudioLM](https://arxiv.org/abs/2209.03143) picks up on this idea to train an audio language model in a purely self-supervised fashion, achieving remarkable performance on audio and speech continuation given a short prompt. On the other hand, [VALL-E](https://arxiv.org/abs/2301.02111) performs speech synthesis via text-conditioned audio language modeling.

## Wrapping up
Despite their original intent, which was to push the boundaries of audio compression while still retaining high percetual quality, Neural Audio Codecs also serve as a crucial building block for modern audio AI models by providing a way to discretize audio into learnable, token-like representations. This tokenization capability has enabled breakthrough models like AudioLM and VALL-E to treat audio generation similarly to how language models handle text generation, opening up exciting possibilities in speech synthesis, audio continuation, speech to speech translation, and other audio-related tasks.

<!-- Here's a table:

| Number | Next number | Previous number |
| :------ |:--- | :--- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One | -->

<!-- {: .box-note}
**Note:** This is a notification box.

{: .box-warning}
**Warning:** This is a warning box.

{: .box-error}
**Error:** This is an error box. -->
