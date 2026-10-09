# OriginX research website

Primary URL: https://robocasa.originxairobotics.com/

Hosted on Cloudflare Pages, project `originx-robotics`, production branch `main`. This is a static HTML/CSS/JS website; there is no build step or runtime secret. The custom subdomain uses a proxied CNAME to `originx-robotics.pages.dev`. The existing apex, www, and other subdomains retain their previous bindings.

Deploy the website folder from the repository root using an authenticated Wrangler installation:

```sh
npx wrangler@4 pages deploy website --project-name originx-robotics --branch main
```

The live site preserves the fixed 1496/2500 benchmark and separately reports 56 strict rescues among 1004 B2000 failures, including all unknown and initial-deviation counts. Model weights remain on Hugging Face and code/evidence on GitHub.
