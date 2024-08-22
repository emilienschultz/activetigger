import { FC } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { useDeleteProject } from '../core/api';
//import { useUserProjects } from '../core/api';
import { useAppContext } from '../core/context';
import { LabelsManagement } from './LabelsManagement';
import { ProjectStatistics } from './ProjectStatistics';
import { SchemesManagement, SelectCurrentScheme } from './forms/SchemesManagementForm';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to display the project page
 */

export const ProjectPage: FC = () => {
  const { projectName } = useParams();

  const {
    appContext: { currentScheme, currentProject: project, reFetchCurrentProject },
  } = useAppContext();

  const availableLabels =
    currentScheme && project ? project.schemes.available[currentScheme] || [] : [];

  const navigate = useNavigate();

  // function to delete project
  const deleteProject = useDeleteProject();
  const actionDelete = () => {
    if (projectName) {
      deleteProject(projectName);
      navigate(`/projects/`);
    }
  };

  return (
    projectName && (
      <ProjectPageLayout projectName={projectName}>
        {project && (
          <div className="container-fluid">
            <div className="row">
              <h2 className="subsection">Project panel</h2>
            </div>

            <div className="row">
              <SelectCurrentScheme></SelectCurrentScheme>
            </div>

            {currentScheme && (
              <div className="row">
                {' '}
                <h4 className="subsection">Statistics of the scheme</h4>
                <ProjectStatistics projectSlug={projectName} scheme={currentScheme} />
              </div>
            )}
            <div className="row mt-4">
              <details className="custom-details">
                <summary className="custom-summary">Parameters of the project</summary>
                <div>{JSON.stringify(project.params, null, 2)}</div>
                <button onClick={actionDelete} className="delete-button">
                  Delete the project
                </button>
              </details>
            </div>
          </div>
        )}
      </ProjectPageLayout>
    )
  );
};
