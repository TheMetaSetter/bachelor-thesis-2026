### [Template] Feature memo: Team invitations
Status: Decided · Owner: Eivind · Researched by: 3 agents · 2026-10-21

[All content are examples, not real.]

## Goal

Let a workspace admin invite colleagues by email, so new users join the right team with the right role.

## Options the research agents found

| Option | Pros | Cons |
| --- | --- | --- |
| A. Email magic-link | Reuses existing auth, simple UX | Links can be forwarded |
| B. Invite codes | No email dependency | Codes leak in Slack, harder to revoke |
| C. SSO domain auto-join | Zero admin work for big customers | Needs SSO on every plan, 3x the scope |

## Decision

**Option A**, with links bound to the invited email and expiring after 72 hours.

Why: it reuses the auth flow we already trust, and binding the link to the email removes the forwarding risk. C is parked until an enterprise customer asks for it.

## What this touches

- `auth/session.ts`: accept an invite token on first sign-in
- `workspaces/<id>/members`: add a `pending` role state
- Data security: the invite never reveals whether an email already has an account

## Open questions

- Should admins see who forwarded a link? (not in v1)
