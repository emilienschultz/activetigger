import { FC } from 'react';
import Tab from 'react-bootstrap/Tab';
import { useNavigate, useParams } from 'react-router-dom';

import { useDeleteProject, useGetLogs } from '../core/api';
//import { useUserProjects } from '../core/api';
import Tabs from 'react-bootstrap/Tabs';
import DataGrid, { Column } from 'react-data-grid';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { AnnotationDisagreementManagement } from './AnnotationDisagreementManagement';
import { ProjectStatistics } from './ProjectStatistics';
import { SchemesManagement } from './SchemesManagement';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to display the project page
 */

interface Row {
  time: string;
  user: string;
  action: string;
}

export const ProjectPage: FC = () => {
  const { projectName } = useParams();

  const {
    appContext: { currentScheme, currentProject: project, history },
    setAppContext,
  } = useAppContext();

  const navigate = useNavigate();

  const { authenticatedUser } = useAuth();

  // get logs
  const { logs } = useGetLogs(projectName || null, authenticatedUser?.username || null, 100);

  // function to delete project
  const deleteProject = useDeleteProject();
  const actionDelete = () => {
    if (projectName) {
      deleteProject(projectName);
      navigate(`/projects/`);
    }
  };

  // function to clear history
  const actionClearHistory = () => {
    setAppContext((prev) => ({ ...prev, history: [] }));
  };

  const activeUsers = project?.users?.active ? project?.users?.active : [];

  const columns: readonly Column<Row>[] = [
    {
      name: 'Time',
      key: 'time',
      resizable: true,
    },
    {
      name: 'User',
      key: 'user',
      resizable: true,
    },
    {
      name: 'Action',
      key: 'action',
    },
  ];

  return (
    projectName && (
      <ProjectPageLayout projectName={projectName}>
        {project && (
          <div className="container-fluid">
            <div className="row">
              <h2 className="subsection">Project panel</h2>
            </div>

            <div className="row">
              <SchemesManagement projectSlug={projectName} />
            </div>

            <Tabs id="panel" className="mb-3" defaultActiveKey="statistics">
              <Tab eventKey="statistics" title="Statistics">
                {currentScheme && (
                  <div className="row">
                    <div className="text-muted smalfont-weight-light">
                      Recent users{' '}
                      {activeUsers.map((e) => (
                        <span className="badge rounded-pill text-bg-light text-muted me-2" key={e}>
                          {e}
                        </span>
                      ))}
                    </div>

                    <ProjectStatistics projectSlug={projectName} scheme={currentScheme} />
                  </div>
                )}
              </Tab>
              <Tab eventKey="disagreement" title="Disagreements">
                {currentScheme && <AnnotationDisagreementManagement projectSlug={projectName} />}
              </Tab>
              <Tab eventKey="session" title="History session">
                <span className="explanations">Element annotated during this session</span>
                <div>{JSON.stringify(history, null, 2)}</div>
                <button onClick={actionClearHistory} className="delete-button">
                  Clear history
                </button>
                <DataGrid<Row>
                  className="fill-grid mt-5"
                  columns={columns}
                  rows={(logs as unknown as Row[]) || []}
                />
              </Tab>
              <Tab eventKey="parameters" title="Parameters">
                <span className="explanations">Parameters of this project</span>
                <div>{JSON.stringify(project.params, null, 2)}</div>
                <button onClick={actionDelete} className="delete-button">
                  Delete project
                </button>
              </Tab>
            </Tabs>
          </div>
        )}
      </ProjectPageLayout>
    )
  );
};
