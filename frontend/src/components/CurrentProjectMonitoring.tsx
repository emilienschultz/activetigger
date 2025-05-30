import { FC, useEffect } from 'react';
import { useParams } from 'react-router-dom';

import { isEqual, isNil } from 'lodash';
import { useProject } from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';

/**
 * Component to actualise project state in the context
 */
export const CurrentProjectMonitoring: FC = () => {
  const { projectName } = useParams();
  const { setAppContext, appContext, resetContext } = useAppContext();
  const { authenticatedUser } = useAuth();

  const { project, reFetch } = useProject(projectName); // api call

  // set a default scheme if there is none
  useEffect(() => {
    const availableSchemes = appContext.currentProject
      ? Object.keys(appContext.currentProject.schemes.available)
      : [];
    // case of there is no selected scheme and schemes are available
    if (!appContext.currentScheme && availableSchemes.length > 0) {
      console.log('Set default scheme');
      setAppContext((state) => ({
        ...state,
        currentScheme: availableSchemes[0],
      }));
    }
  }, [appContext.currentScheme, setAppContext, appContext.currentProject]);

  // reset context if project change
  useEffect(() => {
    if (projectName != appContext.currentProject?.params.project_slug) {
      console.log('Reset context');
      resetContext();
    }
  }, [projectName, appContext.currentProject?.params.project_slug, resetContext]);

  useEffect(() => {
    if (!isNil(project)) {
      // check if training process, and refresh the value
      const isComputing =
        !isNil(authenticatedUser) &&
        !isNil(project.languagemodels.training) &&
        Object.keys(project.languagemodels.training).includes(authenticatedUser.username);

      setAppContext((prev) => {
        if (!isEqual(prev.currentProject, project)) {
          return { ...prev, currentProject: project, isComputing };
        }
        if (prev.isComputing !== isComputing) return { ...prev, isComputing };
        return prev;
      });
    }
  }, [project, setAppContext, authenticatedUser]);

  // Effect to poll project data regularly to monitor long lasting server tasks
  // each time reFetch change
  useEffect(() => {
    //expose refetch method into context
    setAppContext((prev) => ({ ...prev, reFetchCurrentProject: reFetch }));
    // execute a fetch call to update project data every 2000ms
    const intervalId = setInterval(reFetch, 3000);
    // useEffect can return a method which is executed when the component is unmounted
    return () => {
      clearInterval(intervalId);
    };
  }, [reFetch, setAppContext]);

  return null;
};
