import { FC, useState } from 'react';
import { Button } from 'react-bootstrap';
import Modal from 'react-bootstrap/Modal';
import { FaRegTrashAlt } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import { useDeleteProject } from '../core/api';
import { AvailableProjectsModel } from '../types';

interface ProjectCardProps {
  project: AvailableProjectsModel;
}

export const ProjectCard: FC<ProjectCardProps> = ({ project }) => {
  const [showDelete, setShowDelete] = useState(false);
  const handleClose = () => setShowDelete(false);
  const handleShow = () => setShowDelete(true);
  const navigate = useNavigate();

  // function to delete project
  const deleteProject = useDeleteProject();
  const actionDelete = async () => {
    await deleteProject(project.parameters.project_slug);
    handleClose();
    navigate(0);
  };

  // function to navigate to the project page
  const navigateToProject = () => {
    navigate(`/projects/${project.parameters.project_slug}?fromProjectPage=true`);
  };

  return (
    <div
      key={project.parameters.project_slug}
      className="project-card d-flex justify-content-between"
    >
      <div onClick={navigateToProject} className="w-75">
        <h3 className="project-title">{project.parameters.project_name}</h3>
        <p className="project-details">
          <span>Creator: {project.created_by}</span>
          <span>Created at: {project.created_at}</span>
        </p>

        <div className="badge text-bg-info" title="Memory">
          <span className="d-none d-md-inline">HDD : {project.size} Mo</span>
        </div>
      </div>
      <div>
        <div className="trash-wrapper" onClick={handleShow}>
          <FaRegTrashAlt size={20} />
        </div>
        <div>
          <Modal show={showDelete} onHide={handleClose}>
            <Modal.Header>
              <Modal.Title>Delete the project</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              Do you really want to delete the project {project.parameters.project_name}?
            </Modal.Body>
            <Modal.Footer>
              <Button variant="danger" onClick={actionDelete} key="delete">
                Delete
              </Button>
              <Button variant="secondary" onClick={handleClose} key="cancel">
                Exit
              </Button>
            </Modal.Footer>
          </Modal>
        </div>
      </div>
    </div>
  );
};
