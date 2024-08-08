import { FC, useEffect } from 'react';
import { useParams } from 'react-router-dom';

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

  const {
    appContext: { currentScheme, currentProject: project },
    setAppContext,
  } = useAppContext();

  // we update the context with the project currently opened
  useEffect(() => {
    setAppContext((prev) => ({ ...prev, currentProjectSlug: projectName }));
  }, [projectName]);

  return (
    <ProjectPageLayout projectName={projectName}>
      {project && (
        <div>
          <h2>Project panel</h2>
          <div className="explanations">Select a scheme to start annotating</div>
          <div>
            <SchemesManagement
              available_schemes={Object.keys(project.schemes.available)}
              projectSlug={projectName}
            />
          </div>
          {currentScheme && <ProjectStatistics projectSlug={projectName} scheme={currentScheme} />}
        </div>
      )}
    </ProjectPageLayout>
  );
};
