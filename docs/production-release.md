# Production release and email setup

## Email using the existing Gmail address

For the current small deployment, use Gmail SMTP. Enable Google 2-Step
Verification and create an App Password named Examly. Enter that password
directly in Vercel's Production environment as `SMTP_PASSWORD`. Never use the
Google account password, commit credentials, or paste them into chat.

Set the other Production variables:

```dotenv
PUBLIC_BASE_URL=https://mcq-silk.vercel.app
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_STARTTLS=true
SMTP_USERNAME=dinuvishruth123@gmail.com
SMTP_FROM=dinuvishruth123@gmail.com
```

Redeploy after saving. Create a disposable password account with an inbox you
control. Confirm that signup sends a verification link, GET does not consume it,
confirmation verifies the account, password recovery sends a new link, and a
successful reset invalidates both the link and previous sessions. Delete the
disposable account afterward. SMTP acceptance alone does not prove inbox delivery.
For higher volume, move to a transactional provider with an authenticated domain.

References: https://support.google.com/mail/answer/7104828 and
https://developers.google.com/workspace/gmail/imap/imap-smtp

## CI before production

The release workflow runs only after successful push CI on `main`, checks out
that exact commit, and rejects a commit superseded on main. Vercel Git automatic
deployments are disabled in `vercel.json` so they cannot bypass this gate.
Save a Vercel deployment token in GitHub's `production` environment (or repository
Actions secrets) as `VERCEL_TOKEN` before merging these changes. The workflow
uses the existing project and organization IDs, without copying production
application credentials into GitHub. The live deployment keeps serving while
credentials are configured; future automatic releases fail closed without them.

The PostgreSQL test upgrades an old populated schema, preserves an account and
document, simulates historical migration names, runs concurrent boots, and
exercises Settings and account deletion. Never edit an applied migration; add a
new file. Before changes involving existing records, take a database backup or
provider restore point and verify the restoration procedure in isolation.
Application rollback does not reverse a database migration, so schema changes
must remain compatible with the preceding application release.

To roll back application code, use Vercel's rollback to the preceding known good
production deployment. Investigate and fix forward for schema changes.

Reference: https://vercel.com/docs/project-configuration/git-configuration
