import { FC } from 'react';
import { Link } from 'react-router-dom';

import { useUserProjects } from '../core/api';
import { PageLayout } from './layout/PageLayout';

export const ProjectsPage: FC = () => {
  const projects = useUserProjects();

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
            {(projects || []).map((project) => (
              <li key={project.parameters.project_name} className="projects-list">
                <Link to={`/projects/${project.parameters.project_slug}`} className="project-link">
                  <b>{project.parameters.project_name}</b>
                  <br />
                  <p className="project-description">
                    (created by {project.created_by} the {project.created_at})
                  </p>
                </Link>
              </li>
            ))}
          </ul>
        }
      </div>
    </PageLayout>
  );
};
