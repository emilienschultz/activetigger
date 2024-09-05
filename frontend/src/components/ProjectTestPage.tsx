import { FC } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

import { useParams } from 'react-router-dom';
import { useAppContext } from '../core/context';
import { ProjectPageLayout } from './layout/ProjectPageLayout';
import { SelectCurrentScheme } from './SchemesManagement';

/**
 * Test component page
 */
export const ProjectTestPage: FC = () => {
  const { projectName } = useParams();
  //const { authenticatedUser } = useAuth();
  const {
    appContext: { selectionConfig },
    setAppContext,
  } = useAppContext();

  console.log(selectionConfig);

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="test">
      <div className="container-fluid">
        <div className="alert alert-danger m-5">This page is on construction</div>
        <span className="explanations">
          Switch to the test mode to annotate the testset for the selected scheme
        </span>
        <div className="row">
          <div className="col-6">
            <SelectCurrentScheme />
          </div>
          <div className="col-6">X/N of the test dataset annotated</div>
        </div>
        <Tabs id="panel" className="mb-3" defaultActiveKey="annotation">
          <Tab eventKey="annotation" title="1. Annotate">
            <span>- Status of the testset</span>

            <div className="form-check form-switch">
              <input
                className="form-check-input"
                type="checkbox"
                role="switch"
                id="flexSwitchCheckDefault"
                onChange={(e) => {
                  setAppContext((prev) => ({
                    ...prev,
                    selectionConfig: {
                      ...selectionConfig,
                      mode: e.target.checked ? 'test' : 'random',
                    },
                  }));
                }}
                checked={selectionConfig.mode == 'test' ? true : false}
              />
              <label className="form-check-label" htmlFor="flexSwitchCheckDefault">
                Activate test mode
              </label>
            </div>
          </Tab>
          <Tab eventKey="compute" title="2. Compute"></Tab>
        </Tabs>
      </div>
    </ProjectPageLayout>
  );
};
