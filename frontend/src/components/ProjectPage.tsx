import { FC } from 'react';
import { useParams } from 'react-router-dom';

import { useProject } from '../core/api';
import { PageLayout } from './layout/PageLayout';

export const ProjectPage: FC = () => {
  const { projectName } = useParams();

  // API get hook provides the project querying the API for us
  // it also handles auth for us making the component code here very clean
  // project can be undefined has at the very first render the API has not yet responded
  // project undefined means the data is not ready yet or there was an error
  const project = useProject(projectName);

  return (
    <PageLayout currentPage="projects" projectName="test">
      {project && (
        <div>
          <div>
            You are working on the project <span>{project.project_name}</span>
          </div>
          <div>
            <h2>Statistics</h2>
          </div>
          <div>{JSON.stringify(project, null, 2)}</div>
        </div>
      )}
    </PageLayout>
  );
};
