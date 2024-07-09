import { FC, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

import { userProjects } from '../core/api';
import { useAppContext } from '../core/context';
import { PageLayout } from './layout/PageLayout';

// define an object to represent the projects QUESTION : where to put it ?
interface Projects {
  [key: string]: Record<string, never> | undefined;
}

export const ProjectsPage: FC = () => {
  const { appContext } = useAppContext();

  const [projects, setProjects] = useState<Projects>({}); // QUESTION no need to specify the type of element returned ?

  useEffect(() => {
    if (appContext.user?.username) userProjects(appContext.user?.username).then(setProjects);
    else setProjects({});
  }, [appContext.user?.username]);

  return (
    <PageLayout>
      <div className="d-flex justify-content-center align-items-center">
        {
          <ul className="col-md-6">
            <li className="projects-list">
              <Link to="/projects/new" className="project-link new-project">
                Create new project
              </Link>
            </li>
            <li className="projects-title">Existing projects</li>
            {Object.entries(projects).map(([key]) => (
              <li key={key} className="projects-list">
                <Link to={`/projects/${key}`} className="project-link">
                  <b>{key}</b>
                  <br />
                  <p className="project-description">
                    created by {projects[key].created_by} the {projects[key].created_at}
                  </p>
                  {/* QUESTION how to remove this error "Object is possibly 'undefined'"*/}
                </Link>
              </li>
            ))}
          </ul>
        }
      </div>
    </PageLayout>
  );
};
