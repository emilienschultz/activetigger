import { FC } from 'react';
import { useParams } from 'react-router-dom';

import { useDeleteProject } from '../core/api';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

// project deletion page
export const ProjectDeletePage: FC = () => {
  const { projectName } = useParams();

  // we must get the project annotation payload /element
  if (!projectName) return null;

  // function to delete project
  const deleteProject = useDeleteProject();

  return (
    <ProjectPageLayout projectName={projectName} currentAction="delete">
      <div className="container mt-5">
        <div className="row mb-4">
          <div className="col-12"></div>
        </div>
        <div className="row justify-content-center">
          <div className="col-2"></div>
          <div className="col-8 d-flex justify-content-left align-items-center">
            <button
              onClick={() => {
                deleteProject(projectName);
                console.log('Delete project');
              }}
              className="delete-button"
            >
              Validate the deletion of the project
            </button>
          </div>
          <div className="col-2"></div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
