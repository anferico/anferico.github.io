---
layout: post
title: "Neural Audio Codecs & (Residual) Vector Quantization"
subtitle: "The technology behind State-of-the-Art Audio AI models"
cover-img: /assets/img/posts/2023-12-24-neural-audio-codecs/cover.png
thumbnail-img: /assets/img/posts/2023-12-24-neural-audio-codecs/thumb.png
share-img: /assets/img/posts/2023-12-24-neural-audio-codecs/thumb.png
# gh-repo: daattali/beautiful-jekyll
# gh-badge: [star, fork, follow]
tags: [deep-learning, audio]
comments: true
author: Francesco Cariaggi
---

In this blog post, I'll take you through two important concepts behind modern Audio AI models such as Google's [AudioLM](https://arxiv.org/abs/2209.03143) and [VALL-E](https://arxiv.org/abs/2301.02111), Meta's [AudioGen](https://arxiv.org/abs/2209.15352) and [MusicGen](https://arxiv.org/abs/2306.05284), Microsoft's [NaturalSpeech 2](https://arxiv.org/abs/2304.09116) and Suno's [Bark](https://github.com/suno-ai/bark). These concepts are Neural Audio Codecs and (Residual) Vector Quantization.

If you don't mind a short primer/refresher (depending on your prior knowledge) on data compression (needed before delving into the actual topics of this blog post), then just read this blog post from the beginning. Otherwise, if you are already confident with concepts like codecs and bitrate, feel free to skip over to [Neural Audio Codecs](#neural-audio-codecs).

## Introduction

Ever wondered how multimedia such as music and videos are efficiently stored in your PC? Any clue as to how they are transmitted over the internet, e.g. during videocalls?
Even if you don't know the details, you've probably guessed there must be some sort of _compression_ going on at some point. If there was no compression involved, your files would be pretty damn large (ever tried to extract audio tracks from a CD?) and the internet traffic would be much bigger than it currently is.

Let's take audio files as an example. Analog audio is converted to digital form by means of [Pulse-code modulation (PCM)](https://en.wikipedia.org/wiki/Pulse-code_modulation), which simply amounts to sampling the amplitude of analog audio at regular intervals or, equivalently, with a certain _sampling rate_, and then _quantize_ said values to the nearest value within a discrete range, for example 24-bit integers. Now suppose we want to digitize the song [Leyla](https://www.youtube.com/watch?v=TngViNw2pOo) by Derek & The Dominos, which has a duration of 7 minutes and 3 seconds, or equivalently 423 seconds. If we employ a sampling rate of 44.1kHz, our digitized audio will consist of 423 * 44100 = 18654300 samples, each encoded as a 24-bit integer. In this case, the size of the resulting file would be 18654300 * 24 = 447703200 bits = 55962900 bytes = over 53 MiB 🤯.

Nowadays most people listen to music via Spotify and the likes, but you can imagine that prior to that, people couldn't afford to waste tens of MiB of storage (e.g. in their phones) for every single track in their music collection. Indeed, back in the days, music tracks would seldom exceed 2-3 MiB in size (in fact, they could be even smaller than 1 MiB). 

## Codecs
Now we _know_ for a fact that multimedia files get compressed before being stored in a given device or transmitted over the internet, so let's see _how_ the compression takes place. As it turns out, we have software tools called **codecs** (portmanteau of coder/decoder) which serve precisely this purpose. You might not realize, but you have been dealing with codecs all the time: do [MP3](https://en.wikipedia.org/wiki/MP3) and [JPEG](https://en.wikipedia.org/wiki/JPEG) ring a bell? The former is a popular audio codec, whereas the latter is commonly used to compress images.

### Compression parameters
When thinking about compression, a couple of questions arise naturally:
1. How small can I make a file (without making it absolute trash)?
2. How does the compressed file compare to the original file?

The two parameters that help us answer the questions above are **bitrate** and **perceptual quality**.

The bitrate refers to the amount of bits required to encode a "unit" of data. For instance, in the case of audio codecs, said unit of data corresponds to 1 second of audio, hence the bitrate is expressed in _bits per second_ (bps). On the other hand, in the case of image codecs, a unit corresponds to 1 pixel, therefore the bitrate is expressed in _bits per pixel_ (bpp).
Perceptual quality, on the other hand, can be measured either with _objective_ metrics (such as [PESQ](https://en.wikipedia.org/wiki/Perceptual_Evaluation_of_Speech_Quality)) or via _subjective_ evaluations involving expert human listeners.
A good codec aims to _minimize_ the bitrate while _maximizing_ the perceptual quality of the compressed data.

### Codecs categorization
Codecs can be categorized along two orthogonal dimensions:
- **lossy** vs **lossless**
- **generic** vs **content-aware** (I made this term up)

**Lossy** codecs, as the name suggests, are codecs that give up part of the original information to achieve a larger compression rate. Examples of lossy codecs are [MP3](https://en.wikipedia.org/wiki/MP3) for audio and [JPEG](https://en.wikipedia.org/wiki/JPEG) for images.
**Lossless** codecs, on the other hand, are codecs that can retain _all_ the original information while still being able to shrink the data a little bit. Examples of lossless codecs are [FLAC](https://en.wikipedia.org/wiki/FLAC) for audio and [PNG](https://en.wikipedia.org/wiki/PNG) for images.

<div class="box-hands-on" markdown="1">
🤚 **Hands-on**: proof that FLAC is a lossless audio codec

**Step 1**: Get a WAV file off the internet (or find one in your machine)
```sh
wget https://samples-files.com/samples/Audio/wav/sample-file-3.wav -O audio-original.wav
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
In order to achieve the best possible tradeoff between compression rate and perceptual quality of the reconstructed data, traditional audio codecs require careful design using hand-engineered Signal Processing techniques. In situations like this, it is natural to wonder if we can have a neural network *learn* to perform such a complex task from data. As you have probably guessed, the answer is yes, we can indeed do that.

*Neural audio codecs* are neural networks that try to learn how to reconstruct an audio signal given a compressed representation of it. If you're familiar with [Autoencoders](https://en.wikipedia.org/wiki/Autoencoder), this problem formulation won't sound new to you. Unsurprisingly, like autoencoders, neural audio codecs are made of an *encoder* and a *decoder*. The encoder takes a raw audio waveform as input and outputs a compressed representation of it. On the other hand, the decoder is fed the same compressed representation and is tasked with reconstructing the original audio waveform. Although quite simplistic, this description of how a neural audio codec works reveals an important fact about how such models can be trained. More precisely, you might have realized that no human supervision (i.e. labels) is needed to train neural audio codecs! As a matter of fact, the network is essentially requested to learn the identity function: given an audio waveform X, output a reconstructed version X' such that X = X' (in practice, we make it so the two are as close as possible).

Awesome, so basically we're out here racking our brains to train a sophisticated neural architecture to... have it learn the identity function? 😅 That's right, but the point is that we're not interested in the network's output, but rather in the encoder's output, that is the *learned* compressed representation of the input waveform. But how do we obtain that, and how does it even look like? In this blog post, we'll learn about a procedure called **Residual Vector Quantization** which two state-of-the-art neural audio codecs rely on, namely Google's [SoundStream](https://arxiv.org/abs/2107.03312) and Meta's [EnCodec](https://arxiv.org/abs/2210.13438).


## (Residual) Vector Quantization {#residual-vector-quantization}

Broadly speaking, and simply put, quantization is the process through which a continuous representation of a signal is mapped to a discrete space. For instance, as we mentioned at the very beginning of this blog post, Pulse-Code Modulation is a form of quantization.

**Here is some bold text**

## Here is a secondary heading

[This is a link to a different site](https://deanattali.com/) and [this is a link to a section inside this page](#local-urls).

Here's a table:

| Number | Next number | Previous number |
| :------ |:--- | :--- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |

How about a yummy crepe?

![Crepe](https://beautifuljekyll.com/assets/img/crepe.jpg)

It can also be centered!

![Crepe](https://beautifuljekyll.com/assets/img/crepe.jpg){: .mx-auto.d-block :}

Here's a code chunk:

~~~
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~

And here is the same code with syntax highlighting:

```javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

And here is the same code yet again but with line numbers:

{% highlight javascript linenos %}
var foo = function(x) {
  return(x + 5);
}
foo(3)
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.

## Local URLs in project sites {#local-urls}

When hosting a *project site* on GitHub Pages (for example, `https://USERNAME.github.io/MyProject`), URLs that begin with `/` and refer to local files may not work correctly due to how the root URL (`/`) is interpreted by GitHub Pages. You can read more about it [in the FAQ](https://beautifuljekyll.com/faq/#links-in-project-page). To demonstrate the issue, the following local image will be broken **if your site is a project site:**

![Crepe](/assets/img/crepe.jpg)

If the above image is broken, then you'll need to follow the instructions [in the FAQ](https://beautifuljekyll.com/faq/#links-in-project-page). Here is proof that it can be fixed:

![Crepe]({{ '/assets/img/crepe.jpg' | relative_url }})
