import { FC, useState } from 'react';
import Modal from 'react-bootstrap/Modal';
import { useNavigate } from 'react-router-dom';
import { useDeleteProject } from '../core/api';
import { ProjectStateModel } from '../types';
import { ProjectUpdateForm } from './forms/ProjectUpdateForm';
import { ModelParametersTab } from './ModelParametersTab';

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
      <Modal show={showModify} onHide={() => setShowModify(false)} id="addfeature-modal">
        <Modal.Header closeButton>
          <Modal.Title>Add a new feature</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ProjectUpdateForm />
        </Modal.Body>
      </Modal>
      <ModelParametersTab
        params={
          {
            'Project Name': project.params.project_name,
            'Project Slug': project.params.project_slug,
            Filename: project.params.filename,
            'Total Rows': project.params.n_total,
            Language: project.params.language,
            'Text Column': project.params.cols_text,
            'Column ID': project.params.col_id,
            'Colums context': JSON.stringify(project.params.cols_context),
            'Rows in test set': project.params.test ? project.params.n_test : 'Empty',
            'Rows in valid set': project.params.valid ? project.params.n_valid : 'Empty',
          } as Record<string, unknown>
        }
      />

      <div className="horizontal wrap">
        <button className="btn-primary-action" onClick={() => setShowModify(!showModify)}>
          Change parameters
        </button>
        <button onClick={handleShow} className="btn-danger">
          Delete project
        </button>
      </div>

      <div>
        <Modal show={show} onHide={handleClose}>
          <Modal.Header>
            <Modal.Title>Delete the project</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            Do you really want to delete this project
            <div className="horizontal">
              <button onClick={handleClose} style={{ flex: '1 1 auto' }}>
                Cancel
              </button>
              <button className="btn-danger" onClick={actionDelete} style={{ flex: '1 1 auto' }}>
                Confirm
              </button>
            </div>
          </Modal.Body>
        </Modal>
      </div>
    </>
  );
};
