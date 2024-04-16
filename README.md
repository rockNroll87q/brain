# Brain

Brain Imaging and Artificial Intelligence Research lab repo




# Technical Note: Using git subtrees


## TL;DR

1. Add the subtree to your repository (run from the project root):
	1. `git subtree add --prefix=brain git@github.com:rockNroll87q/brain.git main --squash`
2. Commit and push changes normally.
3. When you want to push any code commits to the `brain` repository, run this from your project root:
	1. `git subtree push --prefix=brain git@github.com:rockNroll87q/brain.git <branch_name>`
	2. Note that you may need to change the prefix path from `brain` to some path that keeps in mind your subdirectory structure. For example, in LOD_infant I had to use `LOD-Brain/src/brain` instead.
4. When you want to pull code commits from the `brain` remote on GitHub and update your local repo:
	1. `git subtree pull --prefix=brain git@github.com:rockNroll87q/brain.git <branch_name> --squash`

## Detailed Explanation

Host repository is A, new repository is B.

### Add Repository B as a Subtree

To integrate repository B into A as a subdirectory, use the `git subtree` command:

```bash
git subtree add --prefix=path/to/B <repository-B-URL> main --squash
```

- `--prefix=path/to/B` specifies the path within A where B will be added. This path will be created if it doesn't exist.
- `<repository-B-URL>` is the URL of repository B.
- `main` is the branch from B you wish to integrate. Adjust the branch name as needed.
- `--squash` combines all of B's commits into one, keeping A's history cleaner.

After executing this command, the contents of repository B are now part of A in the specified directory.

### Making Changes in Repository B

Any changes made within repository B (the subtree repository) will now be tracked within the outer repository A. So you can track and commit them as normal:

```bash
cd ../..
git add .
git commit -m "Update subtree B with new changes"
```

#### Pushing to A

When you run the usual `git push` command then the changes reflected in the outer repository are pushed.

```bash
git push origin main
```

### Pushing Subtree Changes Back to Repository B

This is the only tricky part.

If you've made changes to the subtree (repository B) and want to push these back to B's original repository, you'll need to use `git subtree push`. However, this operation can sometimes be complex, especially if there are conflicts or if the structure of repository B doesn't match exactly between the subtree and its original repository.

To push changes from the subtree back to B:

```bash
git subtree push --prefix=path/to/B <repository-B-URL> branch-for-B-changes
```

- `--prefix=path/to/B` is the path to the subtree within A.
- `<repository-B-URL>` is the URL of repository B.
- `branch-for-B-changes` is the branch you want to push your changes to. It's often a good practice to push to a separate branch rather than directly to `main`, to allow for review and integration within repository B's workflow.


## Detailed Example of the `brain` repository with `brain_age`

### Start
So let's assume you have cloned the `brain_age` repository into your local directory, and navigated into it. 
So running `git status` gives something like: 

```bash
On branch main
Your branch is up-to-date with 'origin/main'
```

### Add `brain` subtree

Let's say that we want to add the `brain` repository as a subtree of our `brain_age` one. So within `brain_age`, if we want the new repository to exist inside `./brain` then we will run:

```bash
git subtree add --prefix=brain git@github.com:rockNroll87q/brain.git main --squash
```

The important thing to keep in mind is that this command is run from the **root of the project**. So the prefix path must be relative to the root. If that's 3 layers deep, you need to specify it.
### Changes
Let's pretend that we've made some changes within the `brain` directory. In that case, just like if you've made changes in `brain_age` they'll be tracked, committed, and pushed like normal to `brain_age`. The only tricky part is getting them pushed to `brain` as a separate lib repository.  But even so, to push the changes to the `brain` repository, just run the following command from the `brain_age` root:

```bash
git subtree push --prefix=brain git@github.com:rockNroll87q/brain.git <branch_name>
```

The only part you really need to think about is with what you will replace `<branch_name>`. Let's say you're working on augmentations. Then you might set the branch to `augmentations`. This will do one of a few things:
- If `augmentations` _doesn't yet exist_ in `brain` then the branch is created and any pending commits will be pushed to it, both locally and to the remote (on GitHub).
- If `augmentations` _does exist_ then the commits are collated and pushed to `origin/augmentations`. If there's an active pull request on GitHub, this will be updated.

**Important bit: If you want `brain` updated with your commits, you must run the `git subtree push` separately from your normal push operations

### Updating `brain` from GitHub

If/when brain is updated from GitHub, _or_ you want to switch the branch which `brain` is using, you must run `git subtree pull`. Like this: 

```bash
git subtree pull --prefix=brain git@github.com:rockNroll87q/brain.git <branch_name> --squash
```