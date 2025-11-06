import { FC, useState } from 'react';
import Modal from 'react-bootstrap/Modal';
import { useNavigate } from 'react-router-dom';
import { useDeleteProject } from '../core/api';
import { ProjectStateModel } from '../types';
import { ProjectUpdateForm } from './forms/ProjectUpdateForm';

export interface ProjectParametersModel {
  project: ProjectStateModel;
  projectSlug: string;
}

export const ProjectParameters: FC<ProjectParametersModel> = ({ project, projectSlug }) => {
  const navigate = useNavigate();

  // modals to delete
  const [show, setShow] = useState(false);
  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);

  // show modify
  const [showModify, setShowModify] = useState(false);

  // function to delete project
  const deleteProject = useDeleteProject();
  const actionDelete = async () => {
    if (projectSlug) {
      await deleteProject(projectSlug);
      navigate(`/projects/`);
    }
  };

  return (
    <>
      <div className="explanations">Parameters of this project</div>
      <button className="btn btn-primary" onClick={() => setShowModify(!showModify)}>
        Change parameters
      </button>

      {showModify && <ProjectUpdateForm />}
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
            <td>Total rows</td>
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
            <td>Rows in test set</td>
            <td>{project.params.test ? project.params.n_test : 'empty'}</td>
          </tr>
        </tbody>
      </table>

      <button onClick={handleShow} className="delete-button mt-3">
        Delete project now
      </button>

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
    </>
  );
};
