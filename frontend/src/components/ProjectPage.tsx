import { FC } from 'react';
import { useParams } from 'react-router-dom';

import { useProject, useStatistics } from '../core/api';
import { useAppContext } from '../core/context';
import { FeaturesManagement } from './forms/FeaturesManagementForm';
import { SchemesManagement } from './forms/SchemesManagementForm';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

export const ProjectPage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;

  const {
    appContext: { currentScheme },
  } = useAppContext();

  // API get hook provides the project querying the API for us
  // it also handles auth for us making the component code here very clean
  // project can be undefined has at the very first render the API has not yet responded
  // project undefined means the data is not ready yet or there was an error$

  const { project, reFetch } = useProject(projectName); // get project statefrom the API

  const { statistics } = useStatistics(projectName, currentScheme);

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
          <div>
            <h2>Statistics</h2>
            {JSON.stringify(statistics, null, 2)}
          </div>
        </div>
      )}
    </ProjectPageLayout>
  );
};
