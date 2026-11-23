# Google and GitHub sign-in on Vercel

Use the production domain consistently. Replace `https://YOUR-DOMAIN` below
with the public URL users visit, without a trailing slash.

The confirmed production URL for this application is `https://mcq-silk.vercel.app`.
Its Vercel dashboard project is `dinu-vishruth-mcq` in
`dinu-vishruths-projects`. The existing local `.vercel/repo.json` references
`mcq`, so select the correct project before deploying.

Use these exact production settings:

```text
GOOGLE_REDIRECT_URI=https://mcq-silk.vercel.app/auth/google/callback
GITHUB_REDIRECT_URI=https://mcq-silk.vercel.app/auth/github/callback
```

Register those same callback URLs in the Google and GitHub provider apps.

## Provider applications

In [Google Cloud credentials](https://console.cloud.google.com/apis/credentials),
create an OAuth client of type **Web application**. Add this exact authorized
redirect URI:

```text
https://YOUR-DOMAIN/auth/google/callback
```

Configure the Google Auth Platform audience and consent screen. If the app is
in testing, add the accounts that will test login as test users. Copy the client
ID and client secret into Vercel's server environment variables below.

In [GitHub developer settings](https://github.com/settings/developers), register
an **OAuth App** with homepage `https://YOUR-DOMAIN` and authorization callback:

```text
https://YOUR-DOMAIN/auth/github/callback
```

Generate a client secret and copy the client ID and secret into Vercel. Both
providers use server routes; no client secret belongs in frontend code.

## Vercel environment variables

In the project's **Settings → Environment Variables**, set these for Production:

| Variable | Value |
| --- | --- |
| `SECRET_KEY` | A stable random secret, shared by every function instance |
| `GOOGLE_CLIENT_ID` | Google web OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google web OAuth client secret |
| `GOOGLE_REDIRECT_URI` | `https://YOUR-DOMAIN/auth/google/callback` |
| `GITHUB_CLIENT_ID` | GitHub OAuth App client ID |
| `GITHUB_CLIENT_SECRET` | GitHub OAuth App client secret |
| `GITHUB_REDIRECT_URI` | `https://YOUR-DOMAIN/auth/github/callback` |
| `DATABASE_URL` | Persistent Postgres connection string |

Generate a session secret locally with
`python -c "import secrets; print(secrets.token_hex(32))"` if one is not already
configured. Keep an existing stable secret to preserve active sessions.

Local `.env` values are not deployed to Vercel. Save the production variables,
then redeploy the updated code so Vercel applies them. On startup the app applies
the Postgres migrations, including GitHub identity storage and the unified user
role. Vercel's default `/tmp` SQLite database is temporary and is not suitable
for retaining real accounts across function instances and cold starts.

For previews, either use separate provider apps and registered callback URLs,
or use the production callback configuration. In the latter case, clicking a
provider first sends the browser to the production host before starting OAuth,
so its state cookie is available on return. The production host must already
have the same updated code. Arbitrary preview URLs are not automatically
authorized by Google or GitHub.

## Verify after redeploying

1. Open `/` and `/signup`. Both provider buttons should be working links.
2. Complete Google sign-in; confirm the browser returns to `/dashboard`.
3. Sign out and sign in again; confirm the same account is reused.
4. Repeat with GitHub, including an account with a private email address.
5. Repeat after a fresh deployment to confirm the account persists in Postgres.

If a button says sign-in is unavailable, verify both credentials for that
provider are set in the deployment's environment and redeploy. For a redirect
URI error, compare the production callback URL with the provider registration.
For invalid state, use one browser tab and ensure login starts and finishes on
the same host with a stable `SECRET_KEY`.

Provider references: [Google web-server OAuth](https://developers.google.com/identity/protocols/oauth2/web-server)
and [GitHub OAuth web flow](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps).

## Explicit account linking

Sign-in identifies accounts only by Google subject or GitHub ID. An email match
never attaches a provider to an existing password account: local email fields
are not verified. Signing in with a new provider may therefore create a separate
account, even when the email matches.

To connect a provider to an existing account, sign in to that account and use
Settings → Connected sign-in methods. Password accounts must confirm their current
password; the CSRF-protected POST then starts OAuth and binds its state to that
user ID. The callback refuses a changed signed-in identity or a provider already
attached to another account. Linking must start on the registered callback host.
It never merges learning records or replaces an existing provider identity.

Existing links made by older versions are retained. Review any known suspicious
links before public release; changing the linking rule cannot prove how an old
link was originally established.
