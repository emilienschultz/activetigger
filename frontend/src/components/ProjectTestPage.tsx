import { FC, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

import { useParams } from 'react-router-dom';
import PulseLoader from 'react-spinners/PulseLoader';
import {
  useModelInformations,
  useStatistics,
  useStopTrainBertModel,
  useTestModel,
} from '../core/api';
import { useAppContext } from '../core/context';
import { TestSetCreationForm } from './forms/TestSetCreationForm';
import { ProjectPageLayout } from './layout/ProjectPageLayout';
import { SelectCurrentScheme } from './SchemesManagement';

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

  const kindScheme =
    currentScheme && currentProject
      ? (currentProject.schemes.available[currentScheme]['kind'] as string)
      : 'multiclass';
  // available models
  const availableModels =
    currentScheme && currentProject?.bertmodels.available[currentScheme]
      ? Object.keys(currentProject?.bertmodels.available[currentScheme])
      : [];

  // state forthe model
  const [currentModel, setCurrentModel] = useState<string | null>(null);

  // API hooks
  const { testModel } = useTestModel(
    projectName || null,
    currentScheme || null,
    currentModel || null,
  );
  const { model } = useModelInformations(projectName || null, currentModel || null, isComputing);
  const { stopTraining } = useStopTrainBertModel(projectName || null);

  // get statistics to display (TODO : try a way to avoid another request ?)
  const { statistics } = useStatistics(projectName || null, currentScheme || null);
  if (!projectName) return null;

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="test">
      {kindScheme === 'multilabel' ? (
        <div className="alert alert-info m-2">
          Multi-class testing is not supported for the moment
        </div>
      ) : (
        <div className="container-fluid">
          <div className="explanations">
            Select a scheme and a model, switch to the test mode to annotate the testset, and
            compute test statistics.
          </div>
          {
            // possibility to switch to test mode only if test dataset available
          }
          {currentProject?.params.test && (
            <div>
              <div className="row">
                <div className="col-6">
                  <SelectCurrentScheme />

                  {
                    // Select current model
                    <div className="d-flex align-items-center mb-3">
                      <label
                        htmlFor="model-selected"
                        style={{ whiteSpace: 'nowrap', marginRight: '10px' }}
                      >
                        Current model
                      </label>

                      <select
                        id="model-selected"
                        className="form-select"
                        onChange={(e) => {
                          if (e.target.value) {
                            setCurrentModel(e.target.value);
                          }
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
              </div>
              <Tabs id="panel" className="mb-3" defaultActiveKey="annotation">
                <Tab eventKey="annotation" title="1. Annotate test dataset">
                  <div className="col-6">
                    {statistics && (
                      <span className="badge text-bg-light  m-3">
                        Number of annotations :{' '}
                        {`${statistics['test_annotated_n']} / ${statistics['test_set_n']}`}
                      </span>
                    )}
                  </div>
                  <div className="form-check form-switch">
                    <input
                      className="form-check-input bg-warning"
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
                  {phase == 'test' && (
                    <div className="alert alert-info m-3">
                      Now you can go back to the annotation panel to annotate the test dataset. Once
                      you have a test dataset, you will be able to compute test statistics (next
                      tab. Exit test mode to go back to training dataset.)
                    </div>
                  )}
                </Tab>

                <Tab eventKey="statistics" title="2. Compute statistics">
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
                  {model && model.test_scores ? (
                    <div>
                      <table className="table">
                        <thead>
                          <tr>
                            <th scope="col">Key</th>
                            <th scope="col">Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(model['test_scores']).map(([key, value]) => (
                            <tr key={key}>
                              <td>{key}</td>
                              <td>{JSON.stringify(value)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div>No model selected</div>
                  )}
                </Tab>
              </Tabs>
            </div>
          )}
          {!currentProject?.params.test && (
            <div className="row">
              <div className="col-12">
                <TestSetCreationForm projectSlug={projectName} />
              </div>
            </div>
          )}
        </div>
      )}
    </ProjectPageLayout>
  );
};
