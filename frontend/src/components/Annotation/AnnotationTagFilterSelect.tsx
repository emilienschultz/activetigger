import { FC, useCallback, useMemo } from 'react';
import Select, { MultiValueGenericProps, OptionProps, components } from 'react-select';

import { identity, omit, sortBy, uniq } from 'lodash';
import { useAppContext } from '../../core/context';
import { truncateInMiddle } from '../../core/utils';
import { AnnotationIcon, NoAnnotationIcon, UserIcon } from '../Icons';

export interface TagFilterOption {
  sample: string;
  user?: string | undefined;
  label?: string | undefined;
  value: string;
}

interface AnnotationTagFilterSelectProps {
  availableLabels: string[];
}

function filterOptionValue(option: Omit<TagFilterOption, 'value'>) {
  const value = [option.sample, option.label || '', option.user || ''].join('|');
  return value;
}

/**
 * React select customizations
 **/

const OptionIcon: FC<{ option: TagFilterOption }> = ({ option }) => (
  <>
    {option.sample === 'untagged' && <NoAnnotationIcon className="me-1" />}
    {option.sample === 'tagged' && option.user === undefined && <AnnotationIcon className="me-1" />}
    {option.sample === 'tagged' && option.user !== undefined && <UserIcon className="me-1" />}
  </>
);

const MultiValueLabel = ({ children, ...props }: MultiValueGenericProps<TagFilterOption>) => {
  return (
    <components.MultiValueLabel {...props}>
      <OptionIcon option={props.data} />
      {children}
    </components.MultiValueLabel>
  );
};
const Option = ({ children, ...props }: OptionProps<TagFilterOption>) => {
  return (
    <components.Option {...props}>
      <OptionIcon option={props.data} />
      {children}
    </components.Option>
  );
};

/**
 * Main component
 */
export const AnnotationTagFilterSelect: FC<AnnotationTagFilterSelectProps> = ({
  availableLabels,
}) => {
  // context data
  const {
    appContext: { selectionConfig, currentProject: project },
    setAppContext,
  } = useAppContext();

  // OPTIONS
  const tagFilterOptions: TagFilterOption[] = useMemo(() => {
    const samples =
      // all = no option selected so we remove it from options
      project?.next.sample
        .filter((s) => s !== 'all')
        .map((s) => ({ sample: s, label: undefined, user: undefined })) || [];
    const labels = availableLabels.map((l) => ({
      sample: 'tagged',
      label: l,
      user: undefined,
    }));
    const users =
      project?.users?.users.map((u) => ({
        sample: 'tagged',
        label: undefined,
        user: u,
      })) || [];
    return [...samples, ...labels, ...users].map((o) => ({ ...o, value: filterOptionValue(o) }));
  }, [project?.next.sample, availableLabels, project?.users?.users]);
  // handle option coherence by disabling options depending on current selection Config
  const isTagFilterOptionDisabled = useCallback(
    (option: TagFilterOption) => {
      return (
        selectionConfig.sample !== 'all' &&
        (selectionConfig.sample !== option.sample ||
          //TODO remove this once we have multiple selection
          (option.label !== undefined &&
            selectionConfig.label !== undefined &&
            option.label !== selectionConfig.label) ||
          (option.user !== undefined &&
            selectionConfig.user !== undefined &&
            option.user !== selectionConfig.user))
      );
    },
    [selectionConfig.sample, selectionConfig.label, selectionConfig.user],
  );
  const tagFilterSelectValue = useMemo(() => {
    const selectedValues = [
      filterOptionValue({
        sample: selectionConfig.sample,
        label: selectionConfig.label,
        user: undefined,
      }),
      filterOptionValue({
        sample: selectionConfig.sample,
        label: undefined,
        user: selectionConfig.user,
      }),
    ];
    const values = tagFilterOptions.filter((o) => selectedValues.includes(o.value));
    return values;
  }, [tagFilterOptions, selectionConfig.sample, selectionConfig.label, selectionConfig.user]);

  return (
    <Select
      className="react-select"
      isMulti
      isClearable
      options={tagFilterOptions}
      isOptionDisabled={isTagFilterOptionDisabled}
      // TODO: refacto value after multivalue refacto
      value={tagFilterSelectValue}
      getOptionLabel={(o) =>
        truncateInMiddle(
          o.sample === 'tagged' && o.label !== undefined
            ? o.label
            : o.sample === 'tagged' && o.user !== undefined
              ? `by ${o.user}`
              : o.sample || '',
          20,
        )
      }
      components={{
        MultiValueLabel,
        Option,
      }}
      onChange={(options) => {
        if (options !== null && options.length > 0) {
          // we transform list of options into one selectionConfig object
          const compactOptions: Omit<TagFilterOption, 'value'> = {
            sample: options[0].sample,
            label:
              // TODO remove join after multiple value refacto
              sortBy(uniq(options.map((o) => o.label).filter(identity))).join('|') || undefined,
            user:
              // TODO remove join after multiple value refacto
              sortBy(uniq(options.map((o) => o.user).filter(identity))).join('|') || undefined,
          };
          console.log(compactOptions);
          setAppContext((prev) => {
            return {
              ...prev,
              selectionConfig: {
                ...omit(prev.selectionConfig, ['sample', 'label', 'user']),
                ...compactOptions,
              },
            };
          });
        }
        // if not option selected, selection is all
        else
          setAppContext((prev) => ({
            ...prev,
            selectionConfig: {
              ...omit(prev.selectionConfig, ['sample', 'label', 'user']),
              sample: 'all',
            },
          }));
      }}
    />
  );
};
