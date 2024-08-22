import { FC } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { useDeleteProject } from '../core/api';
//import { useUserProjects } from '../core/api';
import { useAppContext } from '../core/context';
import { LabelsManagement } from './LabelsManagement';
import { ProjectStatistics } from './ProjectStatistics';
import { SchemesManagement } from './forms/SchemesManagementForm';
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

              <details className="custom-details">
                <summary className="custom-summary">Parameters of the project</summary>
                <div>{JSON.stringify(project.params, null, 2)}</div>
              </details>

              <div>
                <h4 className="subsection">Scheme management</h4>

                <SchemesManagement
                  available_schemes={Object.keys(project.schemes.available)}
                  projectSlug={projectName}
                />
              </div>
              <div>
                <h4 className="subsection">Label management</h4>

                {currentScheme && (
                  <div className="d-flex align-items-center">
                    <LabelsManagement
                      projectName={projectName}
                      currentScheme={currentScheme}
                      availableLabels={availableLabels}
                      reFetchCurrentProject={reFetchCurrentProject}
                    />
                  </div>
                )}
              </div>
              {currentScheme && (
                <div>
                  {' '}
                  <h4 className="subsection">Statistics of the scheme</h4>
                  <ProjectStatistics projectSlug={projectName} scheme={currentScheme} />
                </div>
              )}
            </div>
            <div className="row justify-content-left">
              <div className="col-12 d-flex justify-content-center align-items-center">
                <button onClick={actionDelete} className="delete-button m-3">
                  Delete the project
                </button>
              </div>
            </div>
          </div>
        )}
      </ProjectPageLayout>
    )
  );
};
