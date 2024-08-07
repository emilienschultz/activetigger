import { FC } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { useProject } from '../core/api';
import { useDeleteProject } from '../core/api';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

// project deletion page
export const ProjectParametersPage: FC = () => {
  const { projectName } = useParams();

  // we must get the project annotation payload /element
  if (!projectName) return null;

  const { project, reFetch } = useProject(projectName); // get project statefrom the API

  const navigate = useNavigate();

  // function to delete project
  const deleteProject = useDeleteProject();
  const actionDelete = () => {
    deleteProject(projectName);
    console.log('Delete project');
    navigate(`/projects/`);
  };

  return (
    <ProjectPageLayout projectName={projectName} currentAction="parameters">
      <div className="container mt-5">
        <div>
          <h2>Parameters</h2>
          <div>{JSON.stringify(project, null, 2)}</div>
        </div>
        <div className="row mb-4">
          <div className="col-12"></div>
        </div>
        <div className="row justify-content-left">
          <div className="col-8 d-flex justify-content-left align-items-center">
            <button onClick={actionDelete} className="delete-button">
              Delete the project
            </button>
          </div>
          <div className="col-2"></div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
