DEPS_BRANCH="timmy-deps-test"
git checkout master
git restore .
touch .github/stable/doc.txt
pip freeze | grep -v "file:///" > .github/stable/doc.txt
git diff --quiet
if [ $? -eq 0 ]; then
    echo no changes, exiting
    exit
fi
git config user.name "GitHub Actions Bot"
git config user.email "<>"
if git ls-remote --exit-code origin "refs/heads/${DEPS_BRANCH}"; then
    git checkout ${DEPS_BRANCH}
else
    git checkout -b ${DEPS_BRANCH}
fi
git add .github/stable/
git commit -m "Update changed dependencies"
git push -f --set-upstream origin ${DEPS_BRANCH}
open \"$(git ls-remote --get-url $(git config --get branch.$(git branch --show-current).remote) | sed 's|git@github.com:\\(.*\\)$|https://github.com/\\1|' | sed 's|\\.git$||')/compare/$(git config --get branch.$(git branch --show-current).merge | cut -d / -f 3-)?expand=1\"
