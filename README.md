# PhyAI

PhyAI is an Astro site for the PhyAI latency-first serving engine. The site is written in English and has two public sections:

- `/` - the home page, with the pi0.5 quick start, nightly installation command, principles, and latest essay.
- `/about/` - the project positioning and design principles.
- `/blogs/` - the blog index, generated from the `blogs` content collection.
- `/concat/` - links for source code, issues, and documentation.

## Adding an essay

Add a Markdown or MDX file to `src/content/blogs/`. Each essay needs this frontmatter:

```yaml
---
title: "Essay title"
description: "A short summary used on the essay index and in page metadata."
published: 2026-07-16
author: "PhyAI Research"
readTime: "8 min read"
---
```

The filename becomes the URL. For example, `new-system.md` is published at `/blogs/new-system/`.

Use standard Markdown footnotes for optional Inkling-style margin notes. Put the reference beside the sentence it explains and define the note at the end of the file:

```md
Physical context is part of the specification.[^context]

[^context]: The annotation shown beside that sentence.
```

## Local development

```sh
npm install
npm run dev
```

Build the static site with:

```sh
npm run build
```

## Deployment

Pushes to `main` are built and deployed to GitHub Pages by
`.github/workflows/deploy-pages.yml`. In the GitHub repository settings, set
**Pages > Build and deployment > Source** to **GitHub Actions** once before the
first deployment. The workflow also supports manual runs from the Actions tab.

The visual system uses the existing PhyAI brand assets and a compact, responsive editorial layout.
