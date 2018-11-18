# This provides a convienant way to validate all the notebooks.
# NOTE: use python 3.6 when saving the notebooks

# non recursively iterate over notebooks and validate
for i in *.ipynb; do
    py.test --nbval --current-env --disable-warnings "$i"
done
