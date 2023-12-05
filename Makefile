ENVIRONMENT_NAME ?= e3sm-v2-fme
RUN_KEY ?= 42yrs-batch_size8-max_epochs50-lr3e-4-run4
REFERENCE_KEY ?= 42-years-training-set-reference
NOTEBOOKS = daily-precip-pdf daily-precip-zonal-mean-spectrum precip-biases toa-lw-sw-biases wheeler-kiladis

# recommended to deactivate current conda environment before running this
create_environment:
	conda create -n $(ENVIRONMENT_NAME) python=3.8 pip
	conda run -n $(ENVIRONMENT_NAME) pip install -r notebooks/requirements.txt

create_jupyter_kernel: create_environment
	conda run -n $(ENVIRONMENT_NAME) python -m ipykernel install --user \
		--name $(ENVIRONMENT_NAME) --display-name "Python [conda:${ENVIRONMENT_NAME}]"

.PHONY: run_notebooks

run_notebook_%:
	conda env config vars set -n $(ENVIRONMENT_NAME) \
		RUN_KEY=$(RUN_KEY) REFERENCE_KEY=$(REFERENCE_KEY)
	conda run -n $(ENVIRONMENT_NAME) jupyter nbconvert \
		--to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 \
		--ExecutePreprocessor.kernel_name=$(ENVIRONMENT_NAME) \
		--ExecutePreprocessor.allow_errors=True \
		--ExecutePreprocessor.record_timing=True \
		--ExecutePreprocessor.store_widget_state=True \
		--ExecutePreprocessor.iopub_timeout=120 \
		--ExecutePreprocessor.interrupt_on_timeout=True \
		--output-dir=notebooks \
		--output=$*.ipynb \
		notebooks/$*.ipynb

run_notebooks_%: $(addprefix run_notebook_, $(NOTEBOOKS))
