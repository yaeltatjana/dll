#!/bin/bash

output_path="python-wrapper-lib/test/out/"

output_file() {
  local out="${output_path}test_cpp_$1.txt"
  echo "${out}"
}


# to use from dll file
if [ ! -d ${output_path} ]
then
  mkdir ${output_path}
  echo "Create missing output directory "${output_path}
fi

# enter here which reader test you want to launch
# loop inspired by:
#     https://unix.stackexchange.com/questions/450944/bash-loop-through-list-of-strings?answertab=active#tab-top
test_datasets=("mnist" "text")
for t in "${test_datasets[@]}"; do
    ./release/bin/test_datasets "${t}" > "$(output_file "${t}")"
done

# enter here which network test you want to launch
test_networks=("dd" "ddd" "lenet" "alexnet" "vggnet19")
for t in "${test_networks[@]}"; do
    ./release/bin/test_networks "${t}" > "$(output_file "${t}")"
done

