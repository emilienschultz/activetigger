import { FC, useState } from 'react';
import DataGrid, { Column } from 'react-data-grid';
import { SubmitHandler, useForm } from 'react-hook-form';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import PulseLoader from 'react-spinners/PulseLoader';
import { Tooltip } from 'react-tooltip';
import { VictoryAxis, VictoryChart, VictoryLegend, VictoryLine, VictoryTheme } from 'victory';

import { Tab, Tabs } from 'react-bootstrap';
import { FaTools } from 'react-icons/fa';
import {
  useComputeModelPrediction,
  useDeleteBertModel,
  useModelInformations,
  useRenameBertModel,
  useStopTrainBertModel,
  useTrainBertModel,
} from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { newBertModel } from '../types';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to manage model training
 */

interface renameModel {
  new_name: string;
}
interface Row {
  labels: string;
  index: string;
  prediction: string;
  text: string;
}

export const ProjectTrainPage: FC = () => {
  const { projectName: projectSlug } = useParams();

  const { notify } = useNotifications();
  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

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
    currentScheme && project?.bertmodels.available[currentScheme]
      ? Object.keys(project?.bertmodels.available[currentScheme])
      : [];
  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);

  // compute model preduction
  const [batchSize, setBatchSize] = useState<number>(32);
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

  // form to train a model
  const { trainBertModel } = useTrainBertModel(projectSlug || null, currentScheme || null);
  const { stopTraining } = useStopTrainBertModel(projectSlug || null);
  const { handleSubmit: handleSubmitNewModel, register: registerNewModel } = useForm<newBertModel>({
    defaultValues: {
      parameters: {
        batchsize: 4,
        gradacc: 1.0,
        epochs: 3,
        lrate: 5e-5,
        wdecay: 0.01,
        best: true,
        eval: 10,
        gpu: true,
        adapt: false,
      },
    },
  });

  const onSubmitNewModel: SubmitHandler<newBertModel> = async (data) => {
    await trainBertModel(data);
  };

  // loss chart shape data
  const loss = model?.training['loss'] ? JSON.parse(model?.training['loss']) : null;
  const val_epochs = model?.training['loss'] ? Object.values(loss['epoch']) : [];
  const val_loss = model?.training['loss'] ? Object.values(loss['val_loss']) : [];
  const val_eval_loss = model?.training['loss'] ? Object.values(loss['val_eval_loss']) : [];

  const valLossData = val_epochs.map((epoch, i) => ({ x: epoch, y: val_loss[i] }));
  const valEvalLossData = val_epochs.map((epoch, i) => ({ x: epoch, y: val_eval_loss[i] }));

  const LossChart = () => (
    <VictoryChart theme={VictoryTheme.material} minDomain={{ y: 0 }}>
      <VictoryAxis
        label="Epoch"
        style={{
          axisLabel: { padding: 30 },
        }}
      />
      <VictoryAxis
        dependentAxis
        label="Loss"
        style={{
          axisLabel: { padding: 40 },
        }}
      />
      <VictoryLine
        data={valLossData}
        style={{
          data: { stroke: '#c43a31' }, // Rouge pour val_loss
        }}
      />
      <VictoryLine
        data={valEvalLossData}
        style={{
          data: { stroke: '#0000ff' }, // Bleu pour val_eval_loss
        }}
      />
      <VictoryLegend
        x={125}
        y={10}
        title="Legend"
        centerTitle
        orientation="horizontal"
        gutter={20}
        style={{ border: { stroke: 'black' }, title: { fontSize: 10 } }}
        data={[
          { name: 'Loss', symbol: { fill: '#c43a31' } },
          { name: 'Eval Loss', symbol: { fill: '#0000ff' } },
        ]}
      />
    </VictoryChart>
  );

  // display table false prediction
  const falsePredictions = model?.train_scores
    ? (JSON.parse(model.train_scores['false_prediction']) as Row[])
    : null;

  const columns: readonly Column<Row>[] = [
    {
      name: 'Id',
      key: 'index',
      resizable: true,
    },
    {
      name: 'Label',
      key: 'labels',
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

  const displayAdvancement = (val: number | string) => {
    const v = Math.round(Number(val));
    if (v >= 100) return 'completed, please wait';
    return v + '%';
  };

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

              <Tabs id="panel" className="mb-3" defaultActiveKey={'models'}>
                <Tab eventKey="models" title="Models">
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
                  {project?.bertmodels.training &&
                    Object.keys(project.bertmodels.training).length > 0 && (
                      <div className="mt-3">
                        Current training:
                        <ul>
                          {Object.entries(
                            project?.bertmodels.training as Record<
                              string,
                              Record<string, string | number>
                            >,
                          ).map(([_, v]) => (
                            <li key={v.name}>
                              {v.name} - {v.status} :{' '}
                              <span style={{ fontWeight: 'bold' }}>
                                {displayAdvancement(v.progress)}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

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
                                {Object.entries(model.training['parameters']).map(
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
                              <LossChart />
                            </div>
                          </details>
                          <details className="custom-details">
                            <summary>Scores</summary>
                            {!model_scores && !isComputing && (
                              <button
                                className="btn btn-primary me-2 mt-2"
                                onClick={() => computeModelPrediction(currentModel, 'train')}
                              >
                                Predict on training dataset
                              </button>
                            )}
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
                            {model.train_scores && (
                              <div>
                                <table className="table">
                                  {' '}
                                  <thead>
                                    <tr>
                                      <th scope="col">Key</th>
                                      <th scope="col">Value</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    <tr>
                                      <td>F1 micro</td>
                                      <td>{model.train_scores['f1_micro']}</td>
                                    </tr>
                                    <tr>
                                      <td>F1 macro</td>
                                      <td>{model.train_scores['f1_macro']}</td>
                                    </tr>
                                    <tr>
                                      <td>F1 weighted</td>
                                      <td>{model.train_scores['f1_weighted']}</td>
                                    </tr>
                                    <tr>
                                      <td>F1</td>
                                      <td>{JSON.stringify(model.train_scores['f1'])}</td>
                                    </tr>

                                    <tr>
                                      <td>Precision</td>
                                      <td>{JSON.stringify(model.train_scores['precision'])}</td>
                                    </tr>
                                    <tr>
                                      <td>Recall</td>
                                      <td>{JSON.stringify(model.train_scores['recall'])}</td>
                                    </tr>
                                    <tr>
                                      <td>Accuracy</td>
                                      <td>{model.train_scores['accuracy']}</td>
                                    </tr>
                                  </tbody>
                                </table>
                                <details className="m-3">
                                  <summary>False predictions</summary>
                                  {model_scores ? (
                                    <DataGrid
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
                          </details>
                        </div>
                      )}
                    </div>
                  )}
                </Tab>
                <Tab eventKey="new" title="New model">
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
                      <label>Model base</label>

                      <select
                        id="new-model-type"
                        {...registerNewModel('base')}
                        className="form-select"
                      >
                        {(project?.bertmodels.options || []).map((e) => (
                          <option key={e}>{e}</option>
                        ))}
                      </select>
                    </div>

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
                        Epochs{' '}
                        <a className="epochs">
                          <HiOutlineQuestionMarkCircle />
                        </a>
                        <Tooltip anchorSelect=".epochs" place="top">
                          one complete pass through the entire training dataset during the training
                          process
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
                          step size at which the model updates its weights during training
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
                    <div>
                      <label>
                        Eval{' '}
                        <a className="evalstep">
                          <HiOutlineQuestionMarkCircle />
                        </a>
                        <Tooltip anchorSelect=".evalstep" place="top">
                          how often (in terms of training steps) the evaluation of the model on the
                          validation dataset is performed during training
                        </Tooltip>
                      </label>
                      <input type="number" {...registerNewModel('parameters.eval')} />
                    </div>
                    <div className="form-group d-flex align-items-center">
                      <label>Best</label>
                      <input type="checkbox" {...registerNewModel('parameters.best')} />
                    </div>

                    <div className="form-group d-flex align-items-center">
                      <label>GPU</label>
                      <input type="checkbox" {...registerNewModel('parameters.gpu')} />
                    </div>
                    {/* <div className="form-group d-flex align-items-center" style={{ display: 'none' }}>
                    <label>Adapt:</label>
                    <input type="checkbox" {...registerNewModel('parameters.adapt')} />
                  </div> */}
                    {!isComputing && (
                      <button key="start" className="btn btn-primary me-2 mt-2">
                        Train the model
                      </button>
                    )}
                  </form>
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
                </Tab>
                <Tab eventKey="parameters" title="Parameters">
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
                </Tab>
              </Tabs>
            }
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
