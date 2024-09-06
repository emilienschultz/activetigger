import { FC } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

import { useParams } from 'react-router-dom';
import { useStatistics } from '../core/api';
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

  // get statistics to display (TODO : try a way to avoid another request ?)
  const { statistics } = useStatistics(projectName || null, currentScheme || null);
  if (!projectName) return null;
  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="test">
      <div className="container-fluid">
        <div className="alert alert-danger m-5">This page is under construction</div>
        <span className="explanations">
          Switch to the test mode to annotate the testset for the selected scheme
        </span>
        {
          // possibility to switch to test mode only if test dataset available
        }
        {currentProject?.params.test && (
          <div>
            <div className="row">
              <div className="col-6">
                <SelectCurrentScheme />
              </div>
              <div className="col-6">
                {statistics && (
                  <span className="badge text-bg-light  m-3">
                    Number of annotations :{' '}
                    {`${statistics['test_annotated_n']} / ${statistics['test_set_n']}`}
                  </span>
                )}
              </div>
            </div>
            <Tabs id="panel" className="mb-3" defaultActiveKey="annotation">
              <Tab eventKey="annotation" title="1. Annotate">
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
              <Tab eventKey="compute" title="2. Compute"></Tab>
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
