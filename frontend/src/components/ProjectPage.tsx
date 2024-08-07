import { FC, useEffect } from 'react';
import { useParams } from 'react-router-dom';

import { useProject } from '../core/api';
import { useAppContext } from '../core/context';
import { ProjectStatistics } from './ProjectStatistics';
import { FeaturesManagement } from './forms/FeaturesManagementForm';
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

  // API get hook provides the project querying the API for us
  // it also handles auth for us making the component code here very clean
  // project can be undefined has at the very first render the API has not yet responded
  // project undefined means the data is not ready yet or there was an error$

  const { reFetch } = useProject(projectName); // get project statefrom the API

  return (
    <ProjectPageLayout projectName={projectName}>
      {project && (
        <div>
          <div>
            <SchemesManagement
              available_schemes={Object.keys(project.schemes.available)}
              projectSlug={projectName}
              reFetchProject={reFetch}
            />
          </div>
          <div>
            <FeaturesManagement
              projectSlug={projectName}
              reFetchProject={reFetch}
              availableFeatures={project.features.available}
              possibleFeatures={project.features.options}
            />
          </div>
          {currentScheme && <ProjectStatistics projectSlug={projectName} scheme={currentScheme} />}
        </div>
      )}
    </ProjectPageLayout>
  );
};
