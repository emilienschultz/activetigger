import { FC, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ProjectCard } from '../ProjectCard';

import { useUserProjects } from '../../core/api';
import { PageLayout } from '../layout/PageLayout';

import { IoIosAddCircle } from 'react-icons/io';
import { useAppContext } from '../../core/context';
import { AvailableProjectsModel } from '../../types';

export const ProjectsPage: FC = () => {
  // hooks
  const {
    appContext: { currentProject, displayConfig },
    resetContext,
  } = useAppContext();
  const currentProjectSlug = currentProject?.params.project_slug;

  // api call
  const { projects } = useUserProjects();

  // rows to display
  const [rows, setRows] = useState<AvailableProjectsModel[]>([]);
  useEffect(() => {
    setRows(projects || []);
  }, [projects]);

  // handle search input
  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const searchValue = e.target.value.toLowerCase();

    setRows(
      (projects || []).filter((project) => {
        const projectName = project.parameters.project_name.toLowerCase();
        const createdBy = project.created_by.toLowerCase();

        return projectName.includes(searchValue) || createdBy.includes(searchValue);
      }),
    );
  };
  console.log(displayConfig.interfaceType);

  return (
    <PageLayout currentPage="projects">
      <div className="container-fluid">
        {
          <div className="row">
            <div className="col-0 col-lg-3" />
            <div className="col-12 col-lg-6">
              {displayConfig.interfaceType !== 'annotator' && (
                <div className="w-100 d-flex align-items-center justify-content-center">
                  <Link to="/projects/new" className="btn btn-warning w-75 mt-3">
                    <IoIosAddCircle className="m-1" size={30} />
                    Create new project
                  </Link>
                </div>
              )}

              <div className="project-list">
                <input
                  type="text"
                  className="form-control mt-3"
                  placeholder="Search for a project or a user"
                  onChange={handleSearch}
                />
                {rows.map((project) => (
                  <ProjectCard
                    project={project}
                    key={project.parameters.project_slug}
                    resetContext={
                      currentProjectSlug === project.parameters.project_slug
                        ? undefined
                        : resetContext
                    }
                  />
                ))}
              </div>
            </div>
          </div>
        }
      </div>
    </PageLayout>
  );
};
