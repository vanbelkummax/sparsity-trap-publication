# GitHub Repository Setup Guide

This guide will help you create the GitHub repository and push this code.

## Step 1: Create GitHub Repository

### Option A: Via GitHub Web Interface (Recommended)

1. Go to https://github.com/new
2. Fill in the details:
   - **Repository name:** `sparsity-trap-publication` (or your preferred name)
   - **Description:** The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics
   - **Visibility:** Public
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

### Option B: Via GitHub CLI (if installed)

```bash
gh repo create sparsity-trap-publication \
  --public \
  --description "The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics" \
  --source=. \
  --remote=origin
```

## Step 2: Connect Local Repository to GitHub

After creating the GitHub repository, you'll see instructions. Use these commands:

```bash
cd /home/user/sparsity-trap-publication

# Add GitHub as remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/sparsity-trap-publication.git

# Or use SSH (if you have SSH keys set up)
git remote add origin git@github.com:USERNAME/sparsity-trap-publication.git
```

## Step 3: Push to GitHub

```bash
# Push the master branch
git push -u origin master
```

## Step 4: Update Repository URLs

After creating the GitHub repository, you'll need to update the URLs in these files:

### Files to Update

1. **CITATION.cff** - Update repository URL
2. **README.md** - Update clone URL and citation
3. **docs/REPRODUCTION.md** - Update clone URL
4. **CONTRIBUTING.md** - Update clone URL

### Quick Update Script

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
# Set your GitHub username
GITHUB_USER="YOUR_USERNAME"
REPO_NAME="sparsity-trap-publication"

# Update all files
find . -type f \( -name "*.md" -o -name "*.cff" \) -not -path "./.git/*" \
  -exec sed -i "s|vanbelkummax/mse-vs-poisson-2um-benchmark|$GITHUB_USER/$REPO_NAME|g" {} +

find . -type f \( -name "*.md" -o -name "*.cff" \) -not -path "./.git/*" \
  -exec sed -i "s|vanbelkummax/sparsity-trap-publication|$GITHUB_USER/$REPO_NAME|g" {} +

# Commit the changes
git add -A
git commit -m "docs: update GitHub repository URLs"
git push
```

## Step 5: Verify Repository

After pushing, verify at:
- https://github.com/YOUR_USERNAME/sparsity-trap-publication

You should see:
- ✅ README.md displaying with badges
- ✅ CITATION.cff creates "Cite this repository" button
- ✅ LICENSE file recognized
- ✅ All source code and tests
- ✅ Clean file structure

## Step 6: Enable GitHub Features

### Enable Citation Widget
GitHub automatically detects CITATION.cff and adds a "Cite this repository" button in the sidebar.

### Enable Zenodo Integration (for DOI)
1. Go to https://zenodo.org/
2. Sign in with GitHub
3. Enable the repository in Zenodo settings
4. Create a release to trigger DOI generation

### Add Topics (Tags)
Add topics to help people discover your repository:
- `spatial-transcriptomics`
- `deep-learning`
- `loss-functions`
- `poisson-regression`
- `visium-hd`
- `bioinformatics`

## Troubleshooting

### Error: "remote origin already exists"
```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/USERNAME/REPO_NAME.git
```

### Error: "Permission denied (publickey)"
You need to set up SSH keys or use HTTPS with personal access token.
- HTTPS: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
- SSH: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### Error: "Updates were rejected"
```bash
# If you accidentally initialized the GitHub repo with files
git pull origin master --allow-unrelated-histories
git push -u origin master
```

## Next Steps After GitHub Setup

1. ✅ Update README badges with actual URLs
2. ✅ Create first GitHub release (v1.0.0)
3. ✅ Set up GitHub Pages for documentation (optional)
4. ✅ Enable GitHub Actions for CI/CD (optional)
5. ✅ Connect to Zenodo for DOI

## Example: Complete Workflow

```bash
# 1. Create repo on GitHub (via web interface)
# Repository name: sparsity-trap-publication

# 2. Connect local repo
cd /home/user/sparsity-trap-publication
git remote add origin https://github.com/vanbelkummax/sparsity-trap-publication.git

# 3. Push code
git push -u origin master

# 4. URLs are already correct (no changes needed if using vanbelkummax/sparsity-trap-publication)

# 5. Create release
git tag -a v1.0.0 -m "Initial release: Publication-ready package"
git push origin v1.0.0
```

## Contact

If you encounter issues, feel free to reach out:
- Email: max.vanbelkum@vanderbilt.edu
- GitHub: https://github.com/vanbelkummax (once created)
