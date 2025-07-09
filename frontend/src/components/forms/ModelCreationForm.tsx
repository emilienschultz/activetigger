import { FC } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';
import { useTrainBertModel } from '../../core/api';
import { newBertModel, ProjectStateModel } from '../../types';

interface ModelCreationFormProps {
  projectSlug: string | null;
  currentScheme: string | null;
  project: ProjectStateModel | null;
  isComputing: boolean;
}

type BertModel = {
  name: string;
  priority: number;
  comment: string;
  language: string;
};

export const ModelCreationForm: FC<ModelCreationFormProps> = ({
  projectSlug,
  currentScheme,
  project,
  isComputing,
}) => {
  // form to train a model
  const { trainBertModel } = useTrainBertModel(projectSlug || null, currentScheme || null);
  // available base models suited for the project : sorted by language + priority
  const filteredModels = ((project?.languagemodels.options as unknown as BertModel[]) ?? [])
    .sort((a, b) => b.priority - a.priority)
    .sort((a, b) => {
      const aHasFr = a.language === project?.params.language ? -1 : 1;
      const bHasFr = b.language === project?.params.language ? -1 : 1;
      return aHasFr - bHasFr;
    });
  const availableBaseModels = filteredModels.map((e) => ({
    value: e.name as string,
    label: `[${e.language as string}] ${e.name as string}`,
  }));
  // available labels from context
  const availableLabels =
    currentScheme &&
    project &&
    project.schemes.available &&
    project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [];
  const existingLabels = Object.entries(availableLabels).map(([key, value]) => ({
    value: key,
    label: value,
  }));

  const kindScheme =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind
      : 'multiclass';

  const {
    handleSubmit: handleSubmitNewModel,
    register: registerNewModel,
    control,
  } = useForm<newBertModel>({
    defaultValues: {
      class_balance: false,
      class_min_freq: 1,
      test_size: 0.2,
      parameters: {
        batchsize: 4,
        gradacc: 4.0,
        epochs: 3,
        lrate: 3e-5,
        wdecay: 0.01,
        best: true,
        eval: 10,
        gpu: true,
        adapt: false,
      },
    },
  });

  const onSubmitNewModel: SubmitHandler<newBertModel> = async (data) => {
    // setActiveKey('models');
    await trainBertModel(data);
  };
  return (
    <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
      {kindScheme == 'multilabel' && (
        <div role="alert" className="alert alert-warning">
          <label htmlFor="dichotomize">
            This is a multiclass scheme. The model needs to be dichotomize on a specific label
            (yes/no)
          </label>
          <select id="dichotomize" {...registerNewModel('dichotomize')}>
            {Object.values(availableLabels).map((e) => (
              <option key={e}>{e}</option>
            ))}{' '}
          </select>
        </div>
      )}

      <label htmlFor="new-model-type"></label>
      <div>
        <label>Name for the model</label>
        <input
          type="text"
          {...registerNewModel('name')}
          placeholder="Name the model"
          className="form-control"
        />
      </div>

      <div>
        <label>
          Model base{' '}
          <a className="basemodel">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".basemodel" place="top">
            The pre-trained model to be used for fine-tuning.
          </Tooltip>
        </label>

        <Controller
          name="base"
          control={control}
          defaultValue={availableBaseModels?.[0]?.value}
          render={({ field }) => (
            <Select
              {...field}
              options={availableBaseModels}
              classNamePrefix="react-select"
              value={availableBaseModels.find((option) => option.value === field.value)}
              onChange={(selectedOption) => field.onChange(selectedOption?.value)}
            />
          )}
        />
      </div>

      <div>
        <label>
          Epochs{' '}
          <a className="epochs">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".epochs" place="top">
            number of complete pass through the entire training dataset
          </Tooltip>
        </label>
        <input type="number" {...registerNewModel('parameters.epochs')} min={0} />
      </div>
      <div>
        <label>
          Learning Rate{' '}
          <a className="learningrate">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".learningrate" place="top">
            step size at which the model updates its weights during training (use a factor 3 to
            change it)
          </Tooltip>
        </label>
        <input type="number" step="0.00001" min={0} {...registerNewModel('parameters.lrate')} />
      </div>
      <div>
        <label>
          Weight Decay{' '}
          <a className="weightdecay">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".weightdecay" place="top">
            regularization technique that reduces model weights over time to prevent overfitting
          </Tooltip>
        </label>
        <input type="number" step="0.001" min={0} {...registerNewModel('parameters.wdecay')} />
      </div>
      <div className="form-group d-flex align-items-center">
        <label>
          Use GPU
          <a className="gpu">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".gpu" place="top">
            Compute the training on GPU.
          </Tooltip>
        </label>
        <input type="checkbox" {...registerNewModel('parameters.gpu')} />
      </div>
      <details className="custom-details">
        <summary>Advanced parameters for the model</summary>
        <div>
          <label>
            Batch Size{' '}
            <a className="batchsize">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".batchsize" place="top">
              How many samples are processed simultaneously. With small GPU, keep it around 4.
            </Tooltip>
          </label>
          <input type="number" {...registerNewModel('parameters.batchsize')} />
        </div>
        <div>
          <label>
            Gradient Accumulation{' '}
            <a className="gradientacc">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".gradientacc" place="top">
              summing gradients over multiple steps before updating the model weights
            </Tooltip>
          </label>
          <input type="number" step="0.01" {...registerNewModel('parameters.gradacc')} />
        </div>
        <div>
          <label>
            Eval{' '}
            <a className="evalstep">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".evalstep" place="top">
              how often (in terms of training steps) the evaluation of the model on the validation
              dataset is performed during training
            </Tooltip>
          </label>
          <input type="number" {...registerNewModel('parameters.eval')} />
        </div>
        <div>
          <label>
            Validation dataset size{' '}
            <a className="test_size">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".test_size" place="top">
              Eval size for the dev test to compute metrics.
            </Tooltip>
          </label>
          <input type="number" step="0.1" {...registerNewModel('test_size')} />
        </div>
        <div>
          <label>
            Class threshold{' '}
            <a className="class_min_freq">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".class_min_freq" place="top">
              Drop classses with less than this number of elements
            </Tooltip>
          </label>
          <input type="number" step="1" {...registerNewModel('class_min_freq')} />
        </div>
        <div className="form-group d-flex align-items-center">
          <label>
            Balance classes
            <a className="class_balance">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".class_balance" place="top">
              Downsize classes to the lowest one.
            </Tooltip>
          </label>
          <input type="checkbox" {...registerNewModel('class_balance')} />
        </div>
        <div className="form-group d-flex align-items-center">
          <label>
            Keep the best model
            <a className="best">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".best" place="top">
              Keep the model with the lowest validation loss.
            </Tooltip>
          </label>
          <input type="checkbox" {...registerNewModel('parameters.best')} />
        </div>
      </details>
      <details className="custom-details">
        <summary>Advanced parameters for the data</summary>
        <div>
          <label>
            Labels to ignore{' '}
            <a className="ignore">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".ignore" place="top">
              Elements with those labels will be ignored during training
            </Tooltip>
          </label>

          <Controller
            name="exclude_labels"
            control={control}
            render={({ field: { onChange } }) => (
              <Select
                options={existingLabels}
                isMulti
                onChange={(selectedOptions) => {
                  onChange(selectedOptions ? selectedOptions.map((option) => option.label) : []);
                }}
              />
            )}
          />
        </div>
      </details>
      <button key="start" className="btn btn-primary me-2 mt-2" disabled={isComputing}>
        Train the model
      </button>
    </form>
  );
};
