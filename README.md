# Personal Website/Blog

This repository contains the source code for my personal website/blog hosted at [grahamg.dev](https://grahamg.dev). Built using [Hugo](https://gohugo.io/), a fast and modern static site generator, with the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

## Features

- Responsive design using PaperMod theme
- Fast and efficient static site generation
- Built-in search functionality
- Social media integration
- Dark/Light mode support
- SEO-optimized content
- Category and tag support

## Prerequisites

Before you begin, ensure you have the following installed:

- [Hugo Extended](https://gohugo.io/installation/) (v0.80.0 or later)
- [Git](https://git-scm.com/)
- A code editor (VS Code, Sublime Text, etc.)

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/grahamg.dev.git
cd grahamg.dev
```

2. Install theme submodule:
```bash
git submodule init
git submodule update
```

3. Start the Hugo development server:
```bash
hugo server -D
```

4. Open your browser and navigate to `http://localhost:1313`

## Project Structure

```
.
├── archetypes/        # Content template files
├── assets/           # Assets that will be processed by Hugo Pipes
├── content/         # Main content directory
│   ├── posts/      # Blog posts
│   └── pages/      # Static pages
├── layouts/        # Custom layout templates
├── static/         # Static files
├── themes/         # Theme directory
│   └── PaperMod/   # PaperMod theme
├── config.toml     # Hugo configuration file
└── README.md       # This file
```

## Creating Content

To create a new blog post:

```bash
hugo new posts/my-new-post.md
```

The new file will be created in `content/posts/` with pre-populated front matter.

## Deployment

The site is automatically deployed using a CI/CD pipeline when changes are pushed to the main branch. The deployment process:

1. Builds the site using Hugo
2. Optimizes assets (images, CSS, JS)
3. Deploys to production

To build the site manually:

```bash
hugo --minify
```

The built site will be in the `public/` directory.

## Customization

- Site configuration is managed in `config.toml`
- Theme customization can be done in `assets/css/extended/`
- Custom layouts can be added to `layouts/`

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

