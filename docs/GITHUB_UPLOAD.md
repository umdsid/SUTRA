# GitHub upload instructions

1. Unzip this release candidate into a clean directory.
2. Run:
   ```bash
   cd SUTRA_GITHUB_RC1_PART1_PART2
   ./test_release.sh
   ```
3. Confirm the only main-figure products under `part2_paper/main/` are individual panels.
4. Resolve `part2_paper/supplementary/suppfig4/INCOMPLETE_IN_UPLOADED_ARCHIVE.txt` before claiming the supplementary set is complete.
5. Initialize Git only after the test passes:
   ```bash
   git init
   git add .
   git status
   git commit -m "Initial SUTRA reproducibility release"
   ```
6. Create an empty GitHub repository, then add its remote and push:
   ```bash
   git branch -M main
   git remote add origin <YOUR-GITHUB-REPOSITORY>
   git push -u origin main
   ```

Large raw 10x data and Part-1 generated result trees belong outside Git. Publish download/accession information in a data manifest and, where appropriate, frozen result bundles as release/archival assets.
