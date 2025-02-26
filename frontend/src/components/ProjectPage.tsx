import { FC, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { useDeleteProject, useGetLogs } from '../core/api';

import Modal from 'react-bootstrap/Modal';
import DataGrid, { Column } from 'react-data-grid';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import { useAppContext } from '../core/context';
import { ProjectUpdateForm } from './forms/ProjectUpdateForm';
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

  // get logs
  const { logs } = useGetLogs(projectName || null, 100);

  // function to delete project
  const deleteProject = useDeleteProject();
  const actionDelete = async () => {
    if (projectName) {
      await deleteProject(projectName);
      navigate(`/projects/`);
    }
  };

  // function to clear history
  const actionClearHistory = () => {
    setAppContext((prev) => ({ ...prev, history: [] }));
    // setAppContext((prev) => ({ ...prev, selectionHistory: {} }));
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

  // expand trainset
  // const expandTrainSet = useExpandTrainSet(projectName || null);
  // const [nElements, setNElements] = useState<number | undefined>(undefined);

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
                      <a className="recentusers">
                        <HiOutlineQuestionMarkCircle />
                      </a>
                      <Tooltip anchorSelect=".recentusers" place="top">
                        Users who made an action in the project during the last 30 minutes
                      </Tooltip>
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

              <Tab eventKey="session" title="History session">
                <span className="explanations">History of the current session</span>
                <div>
                  Session counter{' '}
                  <a className="history">
                    <HiOutlineQuestionMarkCircle />
                  </a>
                  <Tooltip anchorSelect=".history" place="top">
                    Element annotated during this session. If you annotate already annotated data,
                    it prevents you to see an element twice. Clear it if you want to be able to
                    re-annotate again.
                  </Tooltip>{' '}
                  <span
                    className="badge rounded-pill text-bg-light text-muted me-2"
                    key={history.length}
                  >
                    {history.length}
                  </span>
                </div>
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
                <div className="explanations">Parameters of this project</div>
                <button onClick={handleShow} className="delete-button mt-1">
                  Delete project now
                </button>
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

                <details className="custom-details">
                  <summary>Update project</summary>
                  <ProjectUpdateForm />
                  {/* <div className="col-9 alert alert-warning fw-bold mt-3">
                    <span className="explanations">
                      Add empty elements in the trainset. Be careful, it will erase all features.
                    </span>
                    <div className="d-flex inline my-2">
                      <input
                        type="number"
                        placeholder="Add N"
                        className="input-number mx-2"
                        onChange={(el) => setNElements(Number(el.target.value))}
                        value={nElements}
                      />
                      <button
                        onClick={() => {
                          expandTrainSet(nElements);
                          setNElements(0);
                        }}
                        className="btn btn-primary"
                      >
                        Add elements in trainset
                      </button>
                    </div>
                  </div> */}
                </details>

                <div>
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
                </div>
              </Tab>
            </Tabs>
          </div>
        )}
      </ProjectPageLayout>
    )
  );
};
