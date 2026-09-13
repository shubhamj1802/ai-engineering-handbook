# Deploying the handbook — free, public, permanent

Written for: you, doing this once, in about 30 minutes.

Everything here stays on free tiers. Total cost **$0**, with no credit card required at
any step. The only optional spend is a custom domain (~$10/year) if you ever want one.

| Piece | Service | Free tier | Enough? |
| --- | --- | --- | --- |
| Hosting | Vercel Hobby | 100 GB bandwidth/month, unlimited static pages | Yes, by a wide margin |
| Database | Neon | 0.5 GB storage, always-free project | This app stores ~200 bytes per user |
| Sign-in | Google OAuth | Free, unlimited | Yes |
| Domain | `your-app.vercel.app` | Free | Yes |

> **One restriction to know about:** Vercel's Hobby plan is for non-commercial use. A free
> learning site with no ads and nothing for sale is fine. If you ever monetise it, you move
> to a paid plan or to Cloudflare Pages.

---

## Step 1 — Put the code on GitHub

The repository is already initialised with a first commit. You need to create the remote and
push.

1. Go to <https://github.com/new>
2. Repository name: `ai-engineering-handbook` (or anything you like)
3. Visibility: **Public**
4. **Do not** tick "Add a README", "Add .gitignore" or "Choose a licence" — the repo already
   has all three, and adding them creates a conflict on first push.
5. Click **Create repository**, then run these two commands in the project folder:

```bash
git remote add origin https://github.com/YOUR-USERNAME/ai-engineering-handbook.git
git push -u origin main
```

> **ℹ️ Note — If git asks for a password**
> GitHub stopped accepting account passwords over HTTPS. Either install the
> [GitHub CLI](https://cli.github.com/) and run `gh auth login`, or create a
> [personal access token](https://github.com/settings/tokens) and paste that as the password.

**Before you push, confirm no secrets are going with it:**

```bash
git ls-files | grep -E "^\.env" || echo "clean: no env files tracked"
```

That must print `clean`. `.env.local` is gitignored, but check anyway — a secret pushed to a
public repo is public forever, even after you delete it.

---

## Step 2 — Create the database (Neon)

1. Go to <https://neon.tech> and sign up (GitHub login works, no card needed).
2. **Create project** → any name → pick the region closest to your users.
3. On the dashboard, find **Connection string** and choose the **Pooled connection**.
   It looks like:

   ```
   postgresql://user:password@ep-xxx-pooler.region.aws.neon.tech/neondb?sslmode=require
   ```

4. Copy it. You will paste it into Vercel in step 4.

The `progress` table is created automatically the first time someone saves progress —
there is no migration to run.

> **⚠️ Warning — Use the *pooled* connection string**
> The unpooled one opens a new connection per serverless function invocation and will exhaust
> Neon's connection limit under even light traffic. The pooled host has `-pooler` in it.

---

## Step 3 — Set up Google sign-in for the public

You likely already have a local OAuth client. You are now making it work for everyone.

### 3a. Publish the consent screen

1. [Google Cloud Console](https://console.cloud.google.com/) → your project
2. **APIs & Services → OAuth consent screen**
3. If the publishing status says **Testing**, click **PUBLISH APP** and confirm.

This app requests only `openid`, `email` and `profile`. Those are *non-sensitive* scopes, so
publishing takes effect immediately and **Google does not require a verification review**.
While the app stays in Testing, only the accounts you explicitly list can sign in — at most
100 of them.

You will also need, on the consent screen:

- An **app name** and **user support email** (shown on the Google sign-in dialog)
- A **developer contact email**
- Optionally a homepage, privacy policy and terms URL — not required for non-sensitive scopes

### 3b. Add the production URLs to the OAuth client

**APIs & Services → Credentials →** your OAuth 2.0 Client ID → edit:

| Field | Add this value |
| --- | --- |
| Authorised JavaScript origins | `https://your-app.vercel.app` |
| Authorised redirect URIs | `https://your-app.vercel.app/api/auth/callback/google` |

Keep the existing `http://localhost:3000` entries so local development keeps working.

You will not know your Vercel URL until step 4, so do step 4 first and come back — or use
the project name you plan to use, since Vercel's production URL is
`https://<project-name>.vercel.app`.

> **🚨 Important — The redirect URI must match exactly**
> Character for character, including `https://`, no trailing slash, and the
> `/api/auth/callback/google` path. A mismatch produces Google's `redirect_uri_mismatch`
> error, which is the single most common failure in this whole process.

---

## Step 4 — Deploy to Vercel

1. Go to <https://vercel.com/signup> and sign in **with GitHub**.
2. **Add New… → Project** → find your repository → **Import**.
3. Framework preset is detected as **Next.js**. Leave build settings alone.
4. Expand **Environment Variables** and add these six, before deploying:

| Name | Value |
| --- | --- |
| `AUTH_SECRET` | a fresh random string — generate with the command below |
| `AUTH_URL` | `https://your-app.vercel.app` |
| `NEXT_PUBLIC_SITE_URL` | `https://your-app.vercel.app` |
| `GOOGLE_CLIENT_ID` | from Google Cloud Console |
| `GOOGLE_CLIENT_SECRET` | from Google Cloud Console |
| `DATABASE_URL` | the pooled Neon connection string from step 2 |

Generate a production secret — **do not reuse your local one**:

```bash
node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"
```

5. Click **Deploy**. The first build takes 2–4 minutes (82 lessons, 119 pages).

---

## Step 5 — Verify it actually works

Open your live URL and check each of these:

- [ ] Home page loads, sidebar shows all 29 phases
- [ ] Open any lesson: code blocks are highlighted, Mermaid diagrams render, quizzes respond
- [ ] `⌘K` / `Ctrl+K` opens search and returns results
- [ ] Theme toggle switches dark/light and survives a reload
- [ ] **Sign in with Google** completes and shows your avatar
- [ ] Mark a lesson complete → sign out → sign back in → it is still complete
  *(this is the one that proves Neon is wired up)*
- [ ] Open the site in a different browser, sign in, and confirm the same progress appears
- [ ] `https://your-app.vercel.app/robots.txt` and `/sitemap.xml` both return content

If sign-in fails, the error is almost always one of two things — check the Vercel function
logs (**Deployments → your deployment → Functions**):

| Symptom | Cause | Fix |
| --- | --- | --- |
| `redirect_uri_mismatch` | Google client missing the production callback | Step 3b, exact match |
| `MissingSecret` | `AUTH_SECRET` not set | Add it, then **redeploy** |
| Sign-in works but progress resets | `DATABASE_URL` missing or unpooled | Step 2, pooled string |
| `Access blocked: has not completed verification` | consent screen still in Testing | Step 3a, publish |

> **ℹ️ Note — Environment variables need a redeploy**
> Vercel does not apply new environment variables to an existing deployment. After adding or
> changing one: **Deployments → ⋯ → Redeploy**.

---

## Step 6 — Get it found (optional but worth 10 minutes)

1. [Google Search Console](https://search.google.com/search-console) → add your domain as a
   **URL prefix** property → verify via the HTML tag or DNS.
2. Submit `https://your-app.vercel.app/sitemap.xml`.
3. Indexing takes days to weeks. The sitemap lists all 82 lessons, so they get discovered
   together rather than one at a time.

---

## Updating the site after launch

```bash
git add .
git commit -m "Add lesson on X"
git push
```

That is the whole deployment process from now on. Vercel rebuilds and goes live in a few
minutes. Pull requests get their own preview URL automatically.

Before pushing, run the checks locally:

```bash
npm run verify        # content linter + typecheck + production build
```

---

## A custom domain, if you want one later

1. Buy a domain (Namecheap, Cloudflare Registrar, Porkbun — roughly $10/year).
2. Vercel → your project → **Settings → Domains** → add it, and follow the DNS instructions.
3. Update **three** things or sign-in will break:
   - `AUTH_URL` and `NEXT_PUBLIC_SITE_URL` in Vercel, then redeploy
   - The Google OAuth client's authorised origin and redirect URI
   - Google Search Console property

Vercel provisions the TLS certificate automatically and free.

---

## What this deployment costs at scale

| Monthly visitors | Vercel bandwidth | Neon storage | Cost |
| --- | --- | --- | --- |
| 1,000 | ~2 GB | < 1 MB | $0 |
| 10,000 | ~20 GB | ~2 MB | $0 |
| 50,000 | ~100 GB | ~10 MB | at the Hobby bandwidth limit |

Lesson pages are statically generated at build time, so traffic costs almost nothing — only
`/api/progress` and the auth routes run server-side, and only for signed-in users.

If you ever exceed the free tier, the cheapest next step is Cloudflare Pages (unlimited
bandwidth, free) rather than Vercel Pro.
