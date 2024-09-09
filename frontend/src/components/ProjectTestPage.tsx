import { FC, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

import { useParams } from 'react-router-dom';
import { useModelInformations, useStatistics, useTestModel } from '../core/api';
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
    appContext: { phase, currentScheme, currentProject },
    setAppContext,
  } = useAppContext();

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
  const { model } = useModelInformations(projectName || null, currentModel || null);

  // get statistics to display (TODO : try a way to avoid another request ?)
  const { statistics } = useStatistics(projectName || null, currentScheme || null);
  if (!projectName) return null;
  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="test">
      <div className="container">
        <div className="explanations">
          Select a scheme and a model, switch to the test mode to annotate the testset, and compute
          test statistics
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
                      onChange={(e) => setCurrentModel(e.target.value)}
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
                    className="form-check-input"
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
              </Tab>
              <Tab eventKey="compute" title="2. Compute">
                {currentModel && currentScheme ? (
                  <div className="col-12">
                    <button className="btn btn-primary m-3" onClick={() => testModel()}>
                      Compute the test
                    </button>
                    {
                      // TODO : BLOCK THE BUTTON TO PREVENT MULTIPLE SEND
                    }
                  </div>
                ) : (
                  <div>Select a scheme & a model to start computation</div>
                )}
              </Tab>
              <Tab eventKey="statistics" title="3. Statistics">
                {model ? (
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
    </ProjectPageLayout>
  );
};
