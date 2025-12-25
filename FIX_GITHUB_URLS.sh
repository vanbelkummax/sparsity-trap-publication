#!/bin/bash
# Script to fix all GitHub URLs in the repository
# Usage: bash FIX_GITHUB_URLS.sh YOUR_GITHUB_USERNAME REPO_NAME

if [ $# -ne 2 ]; then
    echo "Usage: bash FIX_GITHUB_URLS.sh YOUR_GITHUB_USERNAME REPO_NAME"
    echo "Example: bash FIX_GITHUB_URLS.sh vanbelkummax sparsity-trap-publication"
    exit 1
fi

GITHUB_USER="$1"
REPO_NAME="$2"

echo "Updating GitHub URLs to: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""

# Update all markdown and cff files
find . -type f \( -name "*.md" -o -name "*.cff" \) -not -path "./.git/*" \
  -exec sed -i "s|vanbelkummax/mse-vs-poisson-2um-benchmark|$GITHUB_USER/$REPO_NAME|g" {} +

find . -type f \( -name "*.md" -o -name "*.cff" \) -not -path "./.git/*" \
  -exec sed -i "s|vanbelkummax/sparsity-trap-publication|$GITHUB_USER/$REPO_NAME|g" {} +

echo "✅ URLs updated in all files"
echo ""
echo "Review changes:"
git diff

echo ""
echo "If everything looks good, commit the changes:"
echo "git add -A"
echo "git commit -m 'docs: update GitHub repository URLs to $GITHUB_USER/$REPO_NAME'"
