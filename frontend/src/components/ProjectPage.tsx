import { FC } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

//import { useUserProjects } from '../core/api';
import { useAppContext } from '../core/context';
import { ProjectStatistics } from './ProjectStatistics';
import { SchemesManagement } from './forms/SchemesManagementForm';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to display the project page
 */

export const ProjectPage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;
  /* TODO check if the project exist, otherwise redirect to the projects page
  const projects = useUserProjects();
  const navigate = useNavigate();
  if (!projects) navigate('/projects');
  else if (!(projectName in projects)) navigate('/projects');*/
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  return (
    <ProjectPageLayout projectName={projectName}>
      {project && (
        <div className="container-fluid">
          <div className="row">
            <h2 className="subsection">Project panel</h2>
            <div className="explanations">Select a scheme to start annotating</div>
            <div>
              <SchemesManagement
                available_schemes={Object.keys(project.schemes.available)}
                projectSlug={projectName}
              />
            </div>
            {currentScheme && (
              <ProjectStatistics projectSlug={projectName} scheme={currentScheme} />
            )}
          </div>
        </div>
      )}
    </ProjectPageLayout>
  );
};
