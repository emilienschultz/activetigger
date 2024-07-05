import { FC, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

import { userProjects } from '../core/api';
import { useAppContext } from '../core/context';
import { PageLayout } from './layout/PageLayout';

export const ProjectsPage: FC = () => {
  const { appContext } = useAppContext();

  const [projects, setProjects] = useState<string[]>([]);

  useEffect(() => {
    if (appContext.user?.username) userProjects(appContext.user?.username).then(setProjects);
    else setProjects([]);
  }, [appContext.user?.username]);

  return (
    <PageLayout>
      <h1>Projects' list</h1>
      <ul>
        {projects.map((p) => (
          <li>
            <Link to={`/projects/${p}`}>{p}</Link>
          </li>
        ))}
      </ul>
    </PageLayout>
  );
};
