## Testing
The tests are located in the `tests/` folder and are composed of shared tests (in `shared_oracles/`) and of personal tests (in `perso-tests/`). Feel free to add your own tests in this last folder!

A file that represents a test sample, has this following format:
```python
0, 1, 2, 3, 4	# abscissa of the points of the real trajectory
0, 0, 1, 1, 0	# ordinate of the points of the real trajectory
0, 1, 2, 3	    # abscissa of the points reported by the location system
3, 3, 2, 2   	# ordinate of the points reported by the location system
# if the file stores the expected error between the trajectories:
9.333			# expected result
0.001           # epsilon: precision of the result
```
## Continuous integration
Please note that the project uses circleCi as a continuous integration tool.

## How to report a bug
Did you find a typo or a bug? Please create an issue in the corresponding GitHub tab.

## Commit Convetions
Commits must be named as follows: `(Type) Description`.

### Commit type
* **Feat**: a new feature
* **Enhance**: to improve an existing feature
* **Perf**: a code change that imporves performance
* **Fix**: a bug fix
* **Fix #[issue]** fix an issue reported on GitHub repo and close the issue
* **Test**: Add missing test or correcting existing tests
* **Config**: change in circleCi configuration file
* **Doc(s)**: Doc only changes
* **Clean**: (re)move some file, folder; formatting, ...
* **Refactor**: refactoring code

### Commit description
Briefly describes the change made and must be in imperative form.

### Before committing
Before commiting, please restart the kernel and clear all outputs of jupyter notebooks.