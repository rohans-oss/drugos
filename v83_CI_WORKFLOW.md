# v83 CI Workflow

An institutional-grade CI workflow was created at `.github/workflows/ci.yml` but the USER-provided PAT does not have the `workflow` scope, so it cannot be pushed through the Git client.

The file was uploaded locally to the branch but had to be removed from the push (due to `scope: workflow` restriction). Add it manually via the GitHub UI or generate a new PAT with the `workflow` scope.

The file contents are available in the local clone at `/home/z/my-project/repo/.github/workflows/ci.yml`.