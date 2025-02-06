---
layout: page
title: Blog posts
subtitle: All my blog posts
---

{% assign posts = site.posts %}

<!-- role="list" needed so that `list-style: none` in Safari doesn't remove the list semantics -->
<ul class="posts-list list-unstyled" role="list">
  {% for post in posts %}
  <li class="post-preview">
    <article>
      <a href="{{ post.url | absolute_url }}">
        <h3>{{ post.title | strip_html }}</h3>

        {% if post.subtitle %}
          <h4 class="post-subtitle">
          {{ post.subtitle | strip_html }}
          </h4>
        {% endif %}
      </a>

      <p class="post-meta">
        <span style="font-style: normal">🗓️</span> {{ post.date | date: "%B %-d, %Y" }}
      </p>

      {% if post.tags.size > 0 %}
      <div class="blog-tags">
        <span>🏷️</span>
        <ul class="d-inline list-inline" role="list">
          {% for tag in post.tags %}
          <li class="list-inline-item">
            <a href="{{ '/tags' | absolute_url }}#{{- tag -}}">{{- tag -}}</a>
          </li>
          {% endfor %}
        </ul>
      </div>
      {% endif %}

    </article>
  </li>
  {% endfor %}
</ul>

