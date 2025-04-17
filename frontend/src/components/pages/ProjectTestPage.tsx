import { FC, useState } from 'react';
import DataGrid, { Column } from 'react-data-grid';

import { Row } from 'react-bootstrap';
import { useNavigate, useParams } from 'react-router-dom';
import PulseLoader from 'react-spinners/PulseLoader';
import {
  useDropTestSet,
  useModelInformations,
  useStatistics,
  useStopTrainBertModel,
  useTestModel,
} from '../../core/api';
import { useAppContext } from '../../core/context';
import { DisplayScores } from '../DisplayScores';
import { TestSetCreationForm } from '../forms/TestSetCreationForm';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';

interface Row {
  id: string;
  label: string;
  prediction: string;
  text: string;
}

/**
 * Component test page
 * - Allow to swich the interface to test mode
 * - Compute prediction & statistics on the testset
 */
export const ProjectTestPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { phase, currentScheme, currentProject, isComputing },
    setAppContext,
  } = useAppContext();
  const navigate = useNavigate();

  const kindScheme =
    currentScheme && currentProject
      ? (currentProject.schemes.available[currentScheme]['kind'] as string)
      : 'multiclass';

  // available models
  const currentModelOptions = currentScheme
    ? currentProject?.languagemodels?.available?.[currentScheme]
    : null;

  const availableModels = currentModelOptions ? Object.keys(currentModelOptions) : [];

  console.log('availableModels', availableModels);

  // state forthe model
  const [currentModel, setCurrentModel] = useState<string | null>(null);

  // API hooks
  const { testModel } = useTestModel(
    projectName || null,
    currentScheme || null,
    currentModel || null,
  );
  const dropTestSet = useDropTestSet(projectName || null);
  const { model } = useModelInformations(projectName || null, currentModel || null, isComputing);
  const { stopTraining } = useStopTrainBertModel(projectName || null);

  // get statistics to display (TODO : try a way to avoid another request ?)
  const { statistics } = useStatistics(projectName || null, currentScheme || null);

  // display table false prediction
  const falsePredictions =
    model?.test_scores && model.test_scores['false_predictions']
      ? model.test_scores['false_predictions']
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

  if (!projectName) return null;

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

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="test">
      {kindScheme === 'multilabel' ? (
        <div className="alert alert-info m-2">
          Multi-class testing is not supported for the moment
        </div>
      ) : (
        <div className="container-fluid">
          <div className="explanations">
            Switch to the test mode to annotate the testset and compute test statistics.
          </div>
          <div className="alert alert-warning col-8">
            Warning: It is important to ensure that the testset does not contaminate the model
            training. To avoid that, do not use statistics from the testset to change trainset
            annotations.
          </div>

          {
            // possibility to switch to test mode only if test dataset available
          }
          {currentProject?.params.test && (
            <div className="row d-flex align-items-center">
              <div className="col-4 form-check form-switch">
                <input
                  className="form-check-input bg-info"
                  type="checkbox"
                  role="switch"
                  id="flexSwitchCheckDefault"
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      phase: e.target.checked ? 'test' : 'train',
                    }));
                  }}
                  checked={phase == 'test' ? true : false}
                />
                <label className="form-check-label" htmlFor="flexSwitchCheckDefault">
                  Activate test mode
                </label>
              </div>
              <div className="col-4">
                {statistics && (
                  <span className="badge text-bg-light  m-3">
                    Test set annotations :{' '}
                    {`${statistics['test_annotated_n']} / ${statistics['test_set_n']}`}
                  </span>
                )}
              </div>
              <div className="col-4">
                {phase != 'test' && (
                  <button
                    className="btn btn-danger"
                    onClick={() => {
                      dropTestSet().then(() => {
                        navigate(`/projects/${projectName}/test`);
                      });
                    }}
                  >
                    Drop existing testset
                  </button>
                )}
              </div>
              {phase == 'test' && (
                <div className="alert alert-info m-3">
                  Now you can go back to the annotation panel to annotate the test dataset. Once you
                  have a test dataset, you will be able to compute test statistics (next tab. Exit
                  test mode to go back to training dataset.)
                </div>
              )}
              <hr></hr>
              <div className="col-6">
                {/* <SelectCurrentScheme /> */}

                {
                  // Select current model
                  <div className="d-flex align-items-center mb-3">
                    <label
                      htmlFor="model-selected"
                      style={{ whiteSpace: 'nowrap', marginRight: '10px' }}
                    >
                      Select model to test
                    </label>

                    <select
                      id="model-selected"
                      className="form-select"
                      onChange={(e) => {
                        setCurrentModel(e.target.value);
                      }}
                    >
                      <option></option>
                      {availableModels.map((e) => (
                        <option key={e}>{e}</option>
                      ))}
                    </select>
                  </div>
                }
              </div>
              {isComputing && (
                <div>You already have a process launched. Wait for it to complete.</div>
              )}
              {currentModel && currentScheme && !isComputing && (
                <div className="col-12">
                  <button
                    className="btn btn-primary m-3"
                    onClick={() => testModel()}
                    disabled={isComputing}
                  >
                    Compute prediction testset
                  </button>
                </div>
              )}
              {currentModel && currentScheme && isComputing && (
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
              {!(currentModel && currentScheme) && (
                <div>Select a scheme & a model to start computation</div>
              )}
              {/* {model && model.test_scores && (
                <div>
                  <table className="table">
                    <thead>
                      <tr>
                        <th scope="col">Key</th>
                        <th scope="col">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {model.test_scores &&
                        Object.entries(model.test_scores)
                          .filter(([key]) => key !== 'false_predictions')
                          .map(([key, value], i) => (
                            <tr key={i}>
                              <td>{key}</td>
                              <td>{JSON.stringify(value)}</td>
                            </tr>
                          ))}
                    </tbody>
                  </table>
                  <details className="m-3">
                    <summary>False predictions</summary>
                    {falsePredictions ? (
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
              )} */}
              {model && model.test_scores && (
                <div>
                  <DisplayScores scores={model.train_scores || {}} />
                  <button onClick={() => downloadModel()}>Download as json</button>
                  <details className="m-3">
                    <summary>False predictions</summary>
                    {falsePredictions ? (
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
            </div>
          )}
          {!currentProject?.params.test && (
            <div className="row">
              <div className="col-12">
                <TestSetCreationForm
                  projectSlug={projectName}
                  currentScheme={currentScheme || null}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </ProjectPageLayout>
  );
};
