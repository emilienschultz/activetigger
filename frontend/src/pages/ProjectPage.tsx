import { FC, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';

import { LabelsManagement } from '../components/LabelsManagement';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { useAppContext } from '../core/context';

import { CodebookDisplay } from '../components/CodeBookDisplay';
import { SchemesManagement } from '../components/SchemesManagement';
import { reorderLabels } from '../core/utils';

/**
 * Component to display the project page
 */
export const ProjectPage: FC = () => {
  // get data
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, displayConfig },
    setAppContext,
  } = useAppContext();

  // define variables
  const kindScheme =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind
      : '';
  const availableLabels = useMemo(
    () =>
      currentScheme && project && project.schemes.available[currentScheme]
        ? project.schemes.available[currentScheme].labels || []
        : [],
    [currentScheme, project],
  );

  // sort labels according to the displayConfig
  const availableLabelsSorted = reorderLabels(
    availableLabels as string[],
    displayConfig.labelsOrder || [],
  );

  // redirect if at least 2 tags

  // get the fact that we come from the create page
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [fromProjectPage, setFromProjectPage] = useState<boolean>(false);
  if (!fromProjectPage && searchParams.get('fromProjectPage') === 'true') {
    setFromProjectPage(true);
  }

  // if conditions, navigate to the tag page
  useEffect(() => {
    if (currentScheme && fromProjectPage && availableLabels.length > 1) {
      navigate(`/projects/${projectSlug}/tag`);
      setFromProjectPage(false);
      console.log('fromProjectPage', fromProjectPage, availableLabels, currentScheme);
    }
  }, [fromProjectPage, availableLabels, navigate, projectSlug, currentScheme]);

  if (!projectSlug || !project) return;

  return (
    <ProjectPageLayout projectName={projectSlug}>
      <div className="container-fluid d-flex justify-content-center">
        <SchemesManagement
          projectSlug={projectSlug}
          deactivateModifications={displayConfig.interfaceType === 'annotator'}
        />
      </div>

      <CodebookDisplay projectSlug={projectSlug} currentScheme={currentScheme || null} />

      {availableLabels.length === 0 && (
        <div className="alert alert-info col-12 mt-2">
          No labels available for this scheme. Please add labels to use this scheme, or create a new
          scheme.
        </div>
      )}

      <LabelsManagement
        projectSlug={projectSlug}
        currentScheme={currentScheme || null}
        availableLabels={availableLabelsSorted}
        kindScheme={kindScheme as string}
        setAppContext={setAppContext}
        deactivateModifications={displayConfig.interfaceType === 'annotator'}
      />
    </ProjectPageLayout>
  );
};
