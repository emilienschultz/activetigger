import { FC, useState } from 'react';
//import DataTable from 'react-data-table-component';
import { SubmitHandler, useForm } from 'react-hook-form';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import { VictoryAxis, VictoryChart, VictoryLegend, VictoryLine, VictoryTheme } from 'victory';

import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import DataGrid, { Column } from 'react-data-grid';

import {
  useComputeModelPrediction,
  useDeleteBertModel,
  useModelInformations,
  useRenameBertModel,
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
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const { model } = useModelInformations(projectSlug || null, currentModel || null);
  const model_scores = model?.train_scores;

  // available models
  const availableModels =
    currentScheme && project?.bertmodels.available[currentScheme]
      ? Object.keys(project?.bertmodels.available[currentScheme])
      : [];
  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);

  // compute model preduction
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null);

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
  const {
    handleSubmit: handleSubmitNewModel,
    register: registerNewModel,
    reset: resetNewModel,
  } = useForm<newBertModel>({
    defaultValues: {
      parameters: {
        batchsize: 4,
        gradacc: 1.0,
        epochs: 3,
        lrate: 5e-5,
        wdecay: 0.01,
        best: true,
        eval: 10,
        gpu: false,
        adapt: true,
      },
    },
  });
  const onSubmitNewModel: SubmitHandler<newBertModel> = async (data) => {
    await trainBertModel(data);
    resetNewModel();
    console.log(data);
  };

  // loss chart shape data
  const loss = model?.training['loss'] ? JSON.parse(model?.training['loss']) : null;
  const val_epochs = model?.training['loss'] ? Object.values(loss['epoch']) : [];
  const val_loss = model?.training['loss'] ? Object.values(loss['val_loss']) : [];
  const val_eval_loss = model?.training['loss'] ? Object.values(loss['val_eval_loss']) : [];

  const valLossData = val_epochs.map((epoch, i) => ({ x: epoch, y: val_loss[i] }));
  const valEvalLossData = val_epochs.map((epoch, i) => ({ x: epoch, y: val_eval_loss[i] }));

  const LossChart = () => (
    <VictoryChart theme={VictoryTheme.material}>
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

  return (
    <ProjectPageLayout projectName={projectSlug || null} currentAction="train">
      <div className="container-fluid">
        <div className="row">
          <div className="col-8">
            <span className="explanations">Train and modify models</span>
            <h4 className="subsection">Existing models</h4>
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
                  }
                }}
              >
                <MdOutlineDeleteOutline size={30} />
              </button>
            </div>

            {currentModel && (
              <div>
                <details className="custom-details">
                  <summary className="custom-summary">Rename</summary>
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
                <details className="custom-details">
                  {' '}
                  <summary className="custom-summary">Description of the model</summary>
                  {model && (
                    <div>
                      {!model_scores && (
                        <button
                          className="btn btn-primary me-2 mt-2"
                          onClick={() => computeModelPrediction(currentModel)}
                        >
                          Compute prediction
                        </button>
                      )}
                      <details>
                        <summary>Parameters of the model</summary>
                        <table className="table">
                          <thead>
                            <tr>
                              <th scope="col">Key</th>
                              <th scope="col">Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(model.training['parameters']).map(([key, value]) => (
                              <tr key={key}>
                                <td>{key}</td>
                                <td>{JSON.stringify(value)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>

                        <LossChart></LossChart>
                      </details>
                      <details>
                        <summary>Scores</summary>
                        {model.train_scores && (
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
                                <td>{String(model.train_scores['f1'])}</td>
                              </tr>
                              <tr>
                                <td>Precision</td>
                                <td>{String(model.train_scores['precision'])}</td>
                              </tr>
                              <tr>
                                <td>Recall</td>
                                <td>{String(model.train_scores['recall'])}</td>
                              </tr>
                              <tr>
                                <td>Accuray</td>
                                <td>{String(model.train_scores['accuracy'])}</td>
                              </tr>
                            </tbody>
                          </table>
                        )}
                      </details>
                      <details>
                        <summary>False predictions</summary>
                        <DataGrid
                          className="fill-grid"
                          columns={columns}
                          rows={falsePredictions || []}
                        />
                      </details>
                    </div>
                  )}
                </details>
              </div>
            )}
            <h4 className="subsection">Train a new model</h4>
            <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
              <label htmlFor="new-model-type"></label>
              <div>
                <label>Model base</label>

                <select id="new-model-type" {...registerNewModel('base')}>
                  {(project?.bertmodels.options || []).map((e) => (
                    <option key={e}>{e}</option>
                  ))}
                </select>
              </div>
              <div>
                <label>Name to identify the model</label>
                <input type="text" {...registerNewModel('name')} placeholder="Name the model" />
              </div>
              <div>
                <label>Batch Size:</label>
                <input type="number" {...registerNewModel('parameters.batchsize')} />
              </div>
              <div>
                <label>Gradient Accumulation:</label>
                <input type="number" step="0.01" {...registerNewModel('parameters.gradacc')} />
              </div>
              <div>
                <label>Epochs:</label>
                <input type="number" {...registerNewModel('parameters.epochs')} />
              </div>
              <div>
                <label>Learning Rate:</label>
                <input type="number" step="0.00001" {...registerNewModel('parameters.lrate')} />
              </div>
              <div>
                <label>Weight Decay:</label>
                <input type="number" step="0.001" {...registerNewModel('parameters.wdecay')} />
              </div>
              <div>
                <label>Eval:</label>
                <input type="number" {...registerNewModel('parameters.eval')} />
              </div>
              <div className="form-group d-flex align-items-center">
                <label>Best:</label>
                <input type="checkbox" {...registerNewModel('parameters.best')} />
              </div>

              <div className="form-group d-flex align-items-center">
                <label>GPU:</label>
                <input type="checkbox" {...registerNewModel('parameters.gpu')} />
              </div>
              <div className="form-group d-flex align-items-center">
                <label>Adapt:</label>
                <input type="checkbox" {...registerNewModel('parameters.adapt')} />
              </div>

              <button className="btn btn-primary me-2 mt-2">Train</button>
            </form>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
