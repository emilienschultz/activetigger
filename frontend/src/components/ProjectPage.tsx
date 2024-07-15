import { FC } from 'react';
import { useParams } from 'react-router-dom';

import { useProject } from '../core/api';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

export const ProjectPage: FC = () => {
  const { projectName } = useParams();

  // API get hook provides the project querying the API for us
  // it also handles auth for us making the component code here very clean
  // project can be undefined has at the very first render the API has not yet responded
  // project undefined means the data is not ready yet or there was an error
  const project = useProject(projectName); // get project

  if (!projectName) return null;
  return (
    <ProjectPageLayout projectName={projectName}>
      {project && (
        <div>
          <div>
            You are working on the project <span>{project.params.project_name}</span>
          </div>
          <div>
            <h2>Schemes</h2>
          </div>
          <div>
            <h2>Statistics</h2>
          </div>
          <div>{JSON.stringify(project, null, 2)}</div>
        </div>
      )}
    </ProjectPageLayout>
  );
};
