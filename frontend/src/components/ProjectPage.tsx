import { FC, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { useDeleteProject, useGetLogs } from '../core/api';

import Modal from 'react-bootstrap/Modal';
import DataGrid, { Column } from 'react-data-grid';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { AnnotationDisagreementManagement } from './AnnotationDisagreementManagement';
import { ProjectPageLayout } from './layout/ProjectPageLayout';
import { ProjectStatistics } from './ProjectStatistics';
import { SchemesManagement } from './SchemesManagement';

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

  // modals to delete
  const [show, setShow] = useState(false);
  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);

  return (
    projectName && (
      <ProjectPageLayout projectName={projectName}>
        {project && (
          <div className="container-fluid">
            <div className="row">
              <SchemesManagement projectSlug={projectName} />
            </div>
            <Tabs id="panel" className="mt-3" defaultActiveKey="statistics">
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
                <span className="explanations">
                  Element annotated during this session. If you annotate already annotated data, it
                  prevents you to see an element twice. Clear it if you want to be able to
                  re-annotate again.
                </span>
                <div>Number of element in history : {history.length}</div>
                <button onClick={actionClearHistory} className="delete-button">
                  Clear history
                </button>
                <div className="subsection">Activity on this project</div>
                <DataGrid<Row>
                  className="fill-grid mt-2"
                  columns={columns}
                  rows={(logs as unknown as Row[]) || []}
                />
              </Tab>
              <Tab eventKey="parameters" title="Parameters">
                <span className="explanations">Parameters of this project</span>
                <table className="table-statistics">
                  <tbody>
                    <tr className="table-delimiter">
                      <td>Parameters</td>
                      <td>Value</td>
                    </tr>
                    <tr>
                      <td>Project name</td>
                      <td>{project.params.project_name}</td>
                    </tr>
                    <tr>
                      <td>Project slug</td>
                      <td>{project.params.project_slug}</td>
                    </tr>
                    <tr>
                      <td>Filename</td>
                      <td>{project.params.filename}</td>
                    </tr>
                    <tr>
                      <td>Total rows file</td>
                      <td>{project.params.n_total}</td>
                    </tr>
                    <tr>
                      <td>Language</td>
                      <td>{project.params.language}</td>
                    </tr>
                    <tr>
                      <td>Columns text</td>
                      <td>{project.params.cols_text}</td>
                    </tr>
                    <tr>
                      <td>Column id</td>
                      <td>{project.params.col_id}</td>
                    </tr>
                    <tr>
                      <td>Colums context</td>
                      <td>{JSON.stringify(project.params.cols_context)}</td>
                    </tr>
                    <tr>
                      <td>Is test dataset</td>
                      <td>{JSON.stringify(project.params.test)}</td>
                    </tr>
                  </tbody>
                </table>
                {/* <div>{JSON.stringify(project.params, null, 2)}</div> */}
                <button onClick={handleShow} className="delete-button mt-5">
                  Delete project now
                </button>
                <Modal show={show} onHide={actionDelete}>
                  <Modal.Header>
                    <Modal.Title>Delete the project</Modal.Title>
                  </Modal.Header>
                  <Modal.Body>Do you really want to delete this project</Modal.Body>
                  <Modal.Footer>
                    <button onClick={handleClose}>No</button>
                    <button onClick={actionDelete}>Delete</button>
                  </Modal.Footer>
                </Modal>
              </Tab>
            </Tabs>
          </div>
        )}
      </ProjectPageLayout>
    )
  );
};
