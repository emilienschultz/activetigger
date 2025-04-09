import { FC, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import DataGrid, { Column } from 'react-data-grid';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { FaTools } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import Select from 'react-select';
import PulseLoader from 'react-spinners/PulseLoader';
import { Tooltip } from 'react-tooltip';
import {
  useComputeModelPrediction,
  useDeleteBertModel,
  useGetServer,
  useModelInformations,
  useRenameBertModel,
  useStopTrainBertModel,
  useTrainBertModel,
} from '../../core/api';
import { useAppContext } from '../../core/context';
import { useNotifications } from '../../core/notifications';
import { newBertModel } from '../../types';
import { DisplayScores } from '../DisplayScores';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';
import { LossChart } from '../vizualisation/lossChart';

/**
 * Component to manage model training
 */

interface renameModel {
  new_name: string;
}

interface Row {
  id: string;
  label: string;
  prediction: string;
  text: string;
}

type BertModel = {
  name: string;
  priority: number;
  comment: string;
  language: string;
};

export const TrainPage: FC = () => {
  const { projectName: projectSlug } = useParams();

  const { notify } = useNotifications();
  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

  const { gpu } = useGetServer(project || null);

  const [activeKey, setActiveKey] = useState<string>('models');

  // available labels from context
  const availableLabels =
    currentScheme && project ? project.schemes.available[currentScheme]['labels'] || [] : [];

  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const { model } = useModelInformations(projectSlug || null, currentModel || null, isComputing);
  const model_scores = model?.train_scores;

  const kindScheme =
    currentScheme && project
      ? (project.schemes.available[currentScheme]['kind'] as string)
      : 'multiclass';

  // available models
  const availableModels =
    currentScheme && project?.languagemodels.available[currentScheme]
      ? Object.keys(project?.languagemodels.available[currentScheme])
      : [];

  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);

  // compute model preduction
  const [batchSize] = useState<number>(32);
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, batchSize);

  // form to rename
  const { renameBertModel } = useRenameBertModel(projectSlug || null);
  const {
    handleSubmit: handleSubmitRename,
    register: registerRename,
    reset: resetRename,
  } = useForm<renameModel>();

  const onSubmitRename: SubmitHandler<renameModel> = async (data) => {
    if (currentModel) {
      await renameBertModel(currentModel, data.new_name);
      resetRename();
    } else notify({ type: 'error', message: 'New name is void' });
  };

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

  // form to train a model
  const { trainBertModel } = useTrainBertModel(projectSlug || null, currentScheme || null);
  const { stopTraining } = useStopTrainBertModel(projectSlug || null);
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
        gpu: gpu ? true : false,
        adapt: false,
      },
    },
  });

  const onSubmitNewModel: SubmitHandler<newBertModel> = async (data) => {
    console.log(data);
    setActiveKey('models');
    console.log(activeKey);
    await trainBertModel(data);
  };

  interface LossData {
    epoch: { [key: string]: number };
    val_loss: { [key: string]: number };
    val_eval_loss: { [key: string]: number };
  }

  // loss chart shape data
  const loss = model?.loss ? (model?.loss as unknown as LossData) : null;

  // display table false prediction
  const falsePredictions =
    model?.train_scores && model.train_scores['false_predictions']
      ? model.train_scores['false_predictions']
      : null;

  const columns: readonly Column<Row>[] = [
    {
      name: 'Id',
      key: 'id',
      resizable: true,
    },
    {
      name: 'Label',
      key: 'label',
      resizable: true,
    },
    {
      name: 'Prediction',
      key: 'prediction',
      resizable: true,
    },
    {
      name: 'Text',
      key: 'text',
      resizable: true,
    },
  ];

  const displayAdvancement = (val: number | string | null) => {
    if (!val) return 'process in the queue waiting to start';
    const v = Math.round(Number(val));
    if (v >= 100) return 'completed, please wait';
    return v + '%';
  };

  const downloadModel = () => {
    if (!model) return; // Ensure model is not null or undefined

    // Convert the model object to a JSON string
    const modelJson = JSON.stringify(model, null, 2);

    // Create a Blob from the JSON string
    const blob = new Blob([modelJson], { type: 'application/json' });

    // Create a temporary link element
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = currentModel || 'model.json';
    link.click();
  };

  console.log(model);

  return (
    <ProjectPageLayout projectName={projectSlug || null} currentAction="train">
      <div className="container-fluid">
        <div className="row">
          <div className="col-8">
            <div className="explanations">
              Train and modify models
              <a className="problems m-2">
                <FaTools />
              </a>
              <Tooltip anchorSelect=".problems" place="top">
                If the model doesn't train, the reason can be the limit of available GPU. Please try
                latter. If the problem persists, contact us.
              </Tooltip>
            </div>
            {
              /* Temporary disable multi-class fine-tuning */

              <Tabs
                id="panel"
                className="mb-3"
                activeKey={activeKey}
                onSelect={(k) => setActiveKey(k || 'models')}
              >
                <Tab eventKey="models" title="Models" onSelect={() => setActiveKey('models')}>
                  <label htmlFor="selected-model">Existing models</label>
                  <div className="d-flex align-items-center">
                    <select
                      id="selected-model"
                      className="form-select"
                      onChange={(e) => setCurrentModel(e.target.value)}
                    >
                      <option></option>
                      {availableModels.map((e) => (
                        <option key={e}>{e}</option>
                      ))}
                    </select>
                    <button
                      className="btn btn p-0"
                      onClick={() => {
                        if (currentModel) {
                          deleteBertModel(currentModel);
                          setCurrentModel(null);
                        }
                      }}
                    >
                      <MdOutlineDeleteOutline size={30} />
                    </button>
                  </div>

                  {/* Display the progress of training models */}
                  {project?.languagemodels.training &&
                    Object.keys(project.languagemodels.training).length > 0 && (
                      <div className="mt-3">
                        Current process:
                        <ul>
                          {Object.entries(
                            project?.languagemodels.training as Record<
                              string,
                              Record<string, string | number | null>
                            >,
                          ).map(([_, v]) => (
                            <li key={v.name}>
                              {v.name} - {v.status} :{' '}
                              <span style={{ fontWeight: 'bold' }}>
                                {displayAdvancement(v.progress)}
                                {
                                  <div className="col-6 col-lg-4">
                                    <LossChart loss={v.loss as unknown as LossData} />
                                  </div>
                                }
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {currentModel && (
                    <div>
                      {model && (
                        <div>
                          <details className="custom-details">
                            <summary>Parameters</summary>
                            <details>
                              <summary>Rename</summary>
                              <form onSubmit={handleSubmitRename(onSubmitRename)}>
                                <input
                                  id="new_name"
                                  className="form-control me-2 mt-2"
                                  type="text"
                                  placeholder="New name of the model"
                                  {...registerRename('new_name')}
                                />
                                <button className="btn btn-primary me-2 mt-2">Rename</button>
                              </form>
                            </details>
                            <table className="table">
                              <thead>
                                <tr>
                                  <th scope="col">Key</th>
                                  <th scope="col">Value</th>
                                </tr>
                              </thead>
                              <tbody>
                                {Object.entries(model.params as Record<string, unknown>).map(
                                  ([key, value]) => (
                                    <tr key={key}>
                                      <td>{key}</td>
                                      <td>{JSON.stringify(value)}</td>
                                    </tr>
                                  ),
                                )}
                              </tbody>
                            </table>
                            <div className="col-6 col-lg-4">
                              <LossChart loss={loss} />
                            </div>
                            {model.valid_scores && (
                              <div>
                                Scores de validation
                                <DisplayScores scores={model.valid_scores} />
                              </div>
                            )}
                          </details>
                          <details className="custom-details">
                            <summary>Scores</summary>

                            {model.train_scores && (
                              <div>
                                <DisplayScores scores={model.train_scores} />
                                <button onClick={() => downloadModel()}>Download as json</button>
                                <details className="m-3">
                                  <summary>False predictions</summary>
                                  {model_scores ? (
                                    <DataGrid<Row>
                                      className="fill-grid"
                                      columns={columns}
                                      rows={falsePredictions || []}
                                    />
                                  ) : (
                                    <div>Compute prediction first</div>
                                  )}
                                </details>
                              </div>
                            )}
                            {!isComputing && currentScheme && (
                              <button
                                className="btn btn-primary m-2 mt-2"
                                onClick={() =>
                                  computeModelPrediction(currentModel, 'train', currentScheme)
                                }
                              >
                                Compute on trainset for statistics
                              </button>
                            )}
                            {isComputing && <div>Computation in progress</div>}
                          </details>
                        </div>
                      )}
                    </div>
                  )}
                </Tab>
                <Tab eventKey="new" title="New model" onSelect={() => setActiveKey('new')}>
                  <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
                    {kindScheme == 'multilabel' && (
                      <div role="alert" className="alert alert-warning">
                        <label htmlFor="dichotomize">
                          This is a multiclass scheme. The model needs to be dichotomize on a
                          specific label (yes/no)
                        </label>
                        <select id="dichotomize" {...registerNewModel('dichotomize')}>
                          {Object.values(availableLabels).map((e) => (
                            <option key={e}>{e}</option>
                          ))}{' '}
                        </select>
                      </div>
                    )}

                    <div className="explanations">
                      A good practice is to have around 50 annotated elements per class before
                      starting the training
                    </div>
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
                            value={availableBaseModels.find(
                              (option) => option.value === field.value,
                            )}
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
                      <input type="number" {...registerNewModel('parameters.epochs')} />
                    </div>
                    <div>
                      <label>
                        Learning Rate{' '}
                        <a className="learningrate">
                          <HiOutlineQuestionMarkCircle />
                        </a>
                        <Tooltip anchorSelect=".learningrate" place="top">
                          step size at which the model updates its weights during training (use a
                          factor 3 to change it)
                        </Tooltip>
                      </label>
                      <input
                        type="number"
                        step="0.00001"
                        {...registerNewModel('parameters.lrate')}
                      />
                    </div>
                    <div>
                      <label>
                        Weight Decay{' '}
                        <a className="weightdecay">
                          <HiOutlineQuestionMarkCircle />
                        </a>
                        <Tooltip anchorSelect=".weightdecay" place="top">
                          regularization technique that reduces model weights over time to prevent
                          overfitting
                        </Tooltip>
                      </label>
                      <input
                        type="number"
                        step="0.001"
                        {...registerNewModel('parameters.wdecay')}
                      />
                    </div>
                    <details className="custom-details">
                      <summary>Advanced parameters</summary>
                      <div>
                        <label>
                          Batch Size{' '}
                          <a className="batchsize">
                            <HiOutlineQuestionMarkCircle />
                          </a>
                          <Tooltip anchorSelect=".batchsize" place="top">
                            How many samples are processed simultaneously. With small GPU, keep it
                            around 4.
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
                        <input
                          type="number"
                          step="0.01"
                          {...registerNewModel('parameters.gradacc')}
                        />
                      </div>
                      <div>
                        <label>
                          Eval{' '}
                          <a className="evalstep">
                            <HiOutlineQuestionMarkCircle />
                          </a>
                          <Tooltip anchorSelect=".evalstep" place="top">
                            how often (in terms of training steps) the evaluation of the model on
                            the validation dataset is performed during training
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
                    </details>
                    {!isComputing && (
                      <button key="start" className="btn btn-primary me-2 mt-2">
                        Train the model
                      </button>
                    )}
                  </form>
                </Tab>
                {/* <Tab
                  eventKey="parameters"
                  title="Parameters"
                  onSelect={() => setActiveKey('parameters')}
                >
                  <div>
                    <label>
                      Batch size for predictions{' '}
                      <a className="batch">
                        <HiOutlineQuestionMarkCircle />
                      </a>
                      <Tooltip anchorSelect=".batch" place="top">
                        Batch used for predict. Keep it small (16 or 32) for small GPU.
                      </Tooltip>
                    </label>
                    <input
                      type="number"
                      step="1"
                      className="m-2 form-control"
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                    />
                  </div>
                </Tab> */}
              </Tabs>
            }
            {isComputing && (
              <div>
                <button
                  key="stop"
                  className="btn btn-primary mt-3 d-flex align-items-center"
                  onClick={stopTraining}
                >
                  <PulseLoader color={'white'} /> Stop current process
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
