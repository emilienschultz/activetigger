import classNames from 'classnames';
import { FC } from 'react';
import { FaCloudDownloadAlt } from 'react-icons/fa';
import { FaListCheck } from 'react-icons/fa6';
import { HiMiniRectangleGroup } from 'react-icons/hi2';
import { IoBookSharp, IoSettingsSharp } from 'react-icons/io5';
import { MdModelTraining, MdOutlineHomeMax } from 'react-icons/md';
import { PiTagDuotone } from 'react-icons/pi';
import { RiAiGenerate } from 'react-icons/ri';
import { Link } from 'react-router-dom';
import { useGetServer } from '../../core/api';
import { useAuth } from '../../core/auth';
import { useNotifications } from '../../core/notifications';
import { ProjectStateModel } from '../../types';
import { ModalErrors } from '../ModalError';
import { PossibleProjectActions } from './ProjectPageLayout';

/* define a component for project action bar 
with the project & the current action*/
export const ProjectActionsSidebar: FC<{
  projectState: ProjectStateModel | null;
  currentProjectAction?: PossibleProjectActions;
  currentMode?: string;
  currentScheme?: string;
  currentUser: string;
  developmentMode?: boolean;
}> = ({
  currentProjectAction,
  projectState,
  currentUser,
  currentScheme,
  //  developmentMode,
}) => {
  const projectName = projectState ? projectState.params.project_slug : null;
  const { authenticatedUser } = useAuth();
  // const nbUsers = projectState ? projectState.users.length : 0;

  // 2 types of menu
  const onlyAnnotator = authenticatedUser?.status === 'annotator';

  // test if computation is currently undergoing
  const currentComputation =
    projectState && projectState.languagemodels
      ? currentUser in projectState.languagemodels.training ||
        currentUser in projectState.simplemodel.training ||
        currentUser in projectState.projections.training ||
        currentUser in projectState.bertopic.training ||
        Object.values(projectState.features.training).length > 0
      : false;

  // display the number of current processes on the server
  const { disk } = useGetServer(projectState || null);

  // notify if disk is full
  const { notify } = useNotifications();
  if (disk ? Number(disk['proportion']) > 98 : false) {
    notify({
      message: 'Disk is almost full, please delete some files or alert the admin',
      type: 'warning',
    });
  }

  const errors = projectState?.errors.map((arr) => arr.join(' - ')) || [];
  const color = '#000000ff';

  return (
    <div className={`project-sidebar d-flex flex-column flex-shrink-0 bg-light`}>
      {!onlyAnnotator && (
        <ul className="nav nav-pills flex-column mb-auto">
          <li className="nav-item  d-none d-md-inline">
            <div
              className="nav-link d-inline-block rounded-pill px-3 py-1 bg-light"
              style={{ lineHeight: '1.1' }}
            >
              <div className="fw-semibold text-dark text-truncate">{projectName}</div>
              <div className="small text-secondary text-truncate" style={{ marginTop: '-2px' }}>
                {currentScheme}
              </div>
            </div>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectName}`}
              className={classNames('nav-link', !currentProjectAction && 'active')}
              aria-current="page"
              title="Access and modify your project parameters"
              style={{ color: color }}
            >
              <IoBookSharp />

              <span className="ms-1">Codebook</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectName}/explore`}
              className={classNames('nav-link', currentProjectAction === 'explore' && 'active')}
              aria-current="page"
              title="Explore your data"
              style={{ color: color }}
            >
              <HiMiniRectangleGroup />
              <span className="ms-1">Explore</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectName}/tag`}
              className={classNames('nav-link', currentProjectAction === 'tag' && 'active')}
              aria-current="page"
              title="Tag your data"
              style={{ color: color }}
            >
              <PiTagDuotone />
              <span className="ms-1">Annotate</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectName}/model`}
              className={classNames('nav-link', currentProjectAction === 'model' && 'active')}
              aria-current="page"
              title="Manage your models"
              style={{ color: color }}
            >
              <MdModelTraining />
              <span className="ms-1">Model</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectName}/validate`}
              className={classNames('nav-link', currentProjectAction === 'validate' && 'active')}
              aria-current="page"
              title="Test your model"
              style={{ color: color }}
            >
              <FaListCheck />

              <span className="ms-1">Evaluate</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectName}/export`}
              className={classNames('nav-link', currentProjectAction === 'export' && 'active')}
              aria-current="page"
              title="Export everything"
              style={{ color: color }}
            >
              <FaCloudDownloadAlt />
              <span className="ms-1">Export</span>
            </Link>
          </li>

          <li className="nav-item">
            <Link
              to={`/projects/${projectName}/generate`}
              className={classNames('nav-link', currentProjectAction === 'generate' && 'active')}
              aria-current="page"
              title="Use generative tools to annotate your data"
              style={{ color: '#820888ff' }}
            >
              <RiAiGenerate />
              <span className="ms-1">Generative</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectName}/settings`}
              className={classNames('nav-link', currentProjectAction === 'settings' && 'active')}
              aria-current="page"
              title="Project settings"
            >
              <IoSettingsSharp />
              <span className="ms-1">Settings</span>
            </Link>
          </li>
          <li className="nav-item ">
            <div className="nav-link">
              <div className="badge text-bg-secondary" title="Memory">
                <span className="d-none d-md-inline">
                  {projectState?.memory ? `${projectState.memory.toFixed(1)} Mo` : ''}
                </span>
              </div>

              <br></br>
              {projectState?.errors && projectState?.errors.length > 0 && (
                <ModalErrors errors={errors} />
              )}
            </div>
          </li>
          {currentComputation && (
            <li className="nav-item ">
              <div className="nav-link">
                <div className="d-flex justify-content-left align-items-center">
                  <div className="spinner-border spinner-border-sm text-warning" role="status">
                    <span className="visually-hidden">Computing</span>
                  </div>
                  <span className="computing d-none d-md-inline">Computing</span>
                </div>
              </div>
            </li>
          )}
        </ul>
      )}
      {onlyAnnotator && (
        <ul className="nav nav-pills flex-column mb-auto">
          <li className="nav-item mt-3">
            <Link
              to={`/projects/${projectName}`}
              className={classNames('nav-link', !currentProjectAction && 'active')}
              aria-current="page"
              title="Project"
            >
              <MdOutlineHomeMax className="m-2" />
              <span>
                <b>{projectName}</b>
              </span>
              <span
                className="mx-2 d-none d-md-inline"
                style={{ fontSize: '0.875rem', color: 'grey' }}
              >
                {currentScheme}
              </span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectName}/tag`}
              className={classNames('nav-link', currentProjectAction === 'tag' && 'active')}
              aria-current="page"
              title="Tag"
            >
              <PiTagDuotone />
              <span>Tag</span>
            </Link>
          </li>
        </ul>
      )}
    </div>
  );
};
